"""Multi-center Prototype Knowledge Distillation compatibility adapter.

The adapter keeps the category-level loss of KD, DKD, or NKD and places it
inside the existing CRLD two-view training protocol. Prototype evidence is
used only for the teacher strong-view same-view target.
"""

import math

import torch
import torch.nn.functional as F

from ._base import Distiller


def _split_output(output):
    if isinstance(output, (tuple, list)):
        logits = output[0]
        info = output[1] if len(output) > 1 else None
    else:
        logits, info = output, None
    feature = info.get("pooled_feat") if isinstance(info, dict) else None
    return logits, feature


def _optional_float(node, name, fallback):
    value = getattr(node, name, None)
    return float(fallback if value is None else value)


def _normalize_probability(probability):
    probability = probability.clamp_min(1e-8)
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _aggregate_prototype_logits(scaled_similarity, aggregation="logmeanexp"):
    aggregation = str(aggregation).lower().replace("-", "")
    if aggregation == "logmeanexp":
        return torch.logsumexp(scaled_similarity, dim=2) - math.log(
            scaled_similarity.size(2)
        )
    if aggregation == "mean":
        return scaled_similarity.mean(dim=2)
    if aggregation == "max":
        return scaled_similarity.max(dim=2).values
    raise ValueError(
        "MPKD.PROTO_AGGREGATION must be one of logmeanexp, mean, or max."
    )


def _kd_from_probability(logits_student, target_probability, temperature):
    log_probability = F.log_softmax(logits_student / temperature, dim=1)
    target_probability = _normalize_probability(target_probability)
    return (
        F.kl_div(log_probability, target_probability, reduction="none")
        .sum(dim=1)
        * (temperature**2)
    )


def _dkd_from_probability(
    logits_student,
    target_probability,
    target,
    alpha,
    beta,
    temperature,
):
    target_probability = _normalize_probability(target_probability)
    student_probability = F.softmax(logits_student / temperature, dim=1)
    class_index = target.reshape(-1, 1)
    gt_mask = torch.zeros_like(student_probability).scatter_(1, class_index, 1.0)
    other_mask = 1.0 - gt_mask

    student_two = torch.cat(
        [
            (student_probability * gt_mask).sum(dim=1, keepdim=True),
            (student_probability * other_mask).sum(dim=1, keepdim=True),
        ],
        dim=1,
    )
    target_two = torch.cat(
        [
            (target_probability * gt_mask).sum(dim=1, keepdim=True),
            (target_probability * other_mask).sum(dim=1, keepdim=True),
        ],
        dim=1,
    )
    tckd = (
        target_two
        * (
            target_two.clamp_min(1e-8).log()
            - student_two.clamp_min(1e-8).log()
        )
    ).sum(dim=1) * (temperature**2)

    student_other = student_probability * other_mask
    target_other = target_probability * other_mask
    student_other = student_other / student_other.sum(dim=1, keepdim=True).clamp_min(1e-8)
    target_other = target_other / target_other.sum(dim=1, keepdim=True).clamp_min(1e-8)
    nckd = (
        target_other
        * (
            target_other.clamp_min(1e-8).log()
            - student_other.clamp_min(1e-8).log()
        )
    ).sum(dim=1) * (temperature**2)
    return alpha * tckd + beta * nckd


def _nkd_from_probability(
    logits_student,
    target_probability,
    target_probability_raw,
    target,
    temperature,
    gamma,
):
    batch_size = target.shape[0]
    target_probability = _normalize_probability(target_probability)
    target_probability_raw = _normalize_probability(target_probability_raw)
    class_index = target.reshape(-1, 1)
    other_mask = torch.ones_like(target_probability).scatter_(1, class_index, 0.0)

    student_log_probability = F.log_softmax(logits_student, dim=1)
    target_term = -(
        target_probability_raw.gather(1, class_index)
        * student_log_probability.gather(1, class_index)
    ).squeeze(1)

    student_other = F.softmax(logits_student / temperature, dim=1) * other_mask
    target_other = target_probability * other_mask
    student_other = student_other / student_other.sum(dim=1, keepdim=True).clamp_min(1e-8)
    target_other = target_other / target_other.sum(dim=1, keepdim=True).clamp_min(1e-8)
    non_target_term = -(
        target_other * student_other.clamp_min(1e-8).log()
    ).sum(dim=1) * (temperature**2)
    return target_term + gamma * non_target_term


def _prototype_gate(
    teacher_probability,
    prototype_probability,
    weak_prediction,
    uncertainty,
    fusion_scale,
    fusion_mode,
    use_uncertainty,
    use_agreement,
    use_positive_gain,
    use_confidence,
):
    anchor_index = weak_prediction.reshape(-1, 1)
    agreement = prototype_probability.argmax(dim=1).eq(weak_prediction).float()
    teacher_anchor = teacher_probability.gather(1, anchor_index).squeeze(1)
    prototype_anchor = prototype_probability.gather(1, anchor_index).squeeze(1)
    positive_gain = (
        (prototype_anchor - teacher_anchor).clamp_min(0.0)
        / (prototype_anchor + 1e-8)
    ).clamp(0.0, 1.0)
    entropy = -(
        prototype_probability
        * prototype_probability.clamp_min(1e-8).log()
    ).sum(dim=1)
    confidence = (
        1.0 - entropy / math.log(prototype_probability.size(1))
    ).clamp(0.0, 1.0)

    gate = torch.full_like(uncertainty, fusion_scale)
    if fusion_mode == "gated":
        if use_uncertainty:
            gate = gate * uncertainty
        if use_agreement:
            gate = gate * agreement
        if use_positive_gain:
            gate = gate * positive_gain
        if use_confidence:
            gate = gate * confidence
    return gate.clamp(0.0, 1.0)


class MPKD(Distiller):
    """KD/DKD/NKD with the CRLD two-view and prototype-target adapter."""

    def __init__(self, student, teacher, cfg, precomputed_prototypes=None):
        super().__init__(student, teacher)
        self.base_method = str(getattr(cfg.MPKD, "BASE_METHOD", "KD")).upper()
        if self.base_method not in {"KD", "DKD", "NKD"}:
            raise ValueError(
                "MPKD.BASE_METHOD must be one of KD, DKD, or NKD; "
                f"received {self.base_method!r}."
            )

        base_cfg = getattr(cfg, self.base_method)
        if self.base_method == "DKD":
            default_ce_weight = base_cfg.CE_WEIGHT
        elif self.base_method == "KD":
            default_ce_weight = base_cfg.LOSS.CE_WEIGHT
        else:
            default_ce_weight = base_cfg.LOSS.CE_WEIGHT
        self.ce_weight = _optional_float(
            cfg.MPKD, "CE_WEIGHT", default_ce_weight
        )
        self.wv_weight = float(getattr(cfg.MPKD, "WV_WEIGHT", 2.0))
        self.cv_weight = float(getattr(cfg.MPKD, "CV_WEIGHT", 0.5))
        self.tau_w = float(getattr(cfg.MPKD, "TAU_W", 0.9))
        self.tau_s = float(getattr(cfg.MPKD, "TAU_S", 0.2))
        self.proto_enabled = bool(getattr(cfg.MPKD, "PROTO_ENABLED", True))

        if self.base_method == "KD":
            self.temperature = _optional_float(
                cfg.MPKD, "TEMPERATURE", base_cfg.TEMPERATURE
            )
            self.distill_weight = _optional_float(
                cfg.MPKD, "KD_WEIGHT", base_cfg.LOSS.KD_WEIGHT
            )
        elif self.base_method == "DKD":
            self.temperature = _optional_float(cfg.MPKD, "TEMPERATURE", base_cfg.T)
            self.alpha = _optional_float(cfg.MPKD, "ALPHA", base_cfg.ALPHA)
            self.beta = _optional_float(cfg.MPKD, "BETA", base_cfg.BETA)
            self.warmup = max(
                1.0, _optional_float(cfg.MPKD, "WARMUP", base_cfg.WARMUP)
            )
        else:
            self.temperature = _optional_float(cfg.MPKD, "TEMPERATURE", base_cfg.T)
            self.gamma = _optional_float(cfg.MPKD, "GAMMA", base_cfg.GAMMA)
            self.distill_weight = _optional_float(
                cfg.MPKD, "KD_WEIGHT", base_cfg.LOSS.KD_WEIGHT
            )

        if self.temperature <= 0.0:
            raise ValueError("MPKD distillation temperature must be positive.")
        if min(self.ce_weight, self.wv_weight, self.cv_weight) < 0.0:
            raise ValueError("MPKD loss weights must be non-negative.")

        self.proto_temp = float(getattr(cfg.MPKD, "PROTO_TEMP", 0.5))
        self.proto_aggregation = str(
            getattr(cfg.MPKD, "PROTO_AGGREGATION", "logmeanexp")
        ).lower().replace("-", "")
        if self.proto_aggregation not in {"logmeanexp", "mean", "max"}:
            raise ValueError(
                "MPKD.PROTO_AGGREGATION must be one of logmeanexp, mean, or max."
            )
        self.proto_fusion_scale = float(
            getattr(cfg.MPKD, "PROTO_FUSION_SCALE", 0.5)
        )
        self.proto_fusion_mode = str(
            getattr(cfg.MPKD, "PROTO_FUSION_MODE", "gated")
        ).lower()
        if self.proto_temp <= 0.0:
            raise ValueError("MPKD.PROTO_TEMP must be positive.")
        if not 0.0 <= self.proto_fusion_scale <= 1.0:
            raise ValueError("MPKD.PROTO_FUSION_SCALE must be in [0, 1].")
        if self.proto_fusion_mode not in {"fixed", "gated"}:
            raise ValueError("MPKD.PROTO_FUSION_MODE must be 'fixed' or 'gated'.")

        self.gate_uncertainty = bool(
            getattr(cfg.MPKD, "PROTO_GATE_UNCERTAINTY", True)
        )
        self.gate_agreement = bool(
            getattr(cfg.MPKD, "PROTO_GATE_AGREEMENT", True)
        )
        self.gate_positive_gain = bool(
            getattr(cfg.MPKD, "PROTO_GATE_POSITIVE_GAIN", True)
        )
        self.gate_confidence = bool(
            getattr(cfg.MPKD, "PROTO_GATE_CONFIDENCE", True)
        )

        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

        if self.proto_enabled:
            if precomputed_prototypes is None:
                raise ValueError(
                    "MPKD.PROTO_ENABLED=True requires precomputed prototypes."
                )
            self.max_entropy = math.log(precomputed_prototypes.size(0))
            self.register_buffer("prototypes", precomputed_prototypes.detach().clone())
            self.prototypes = F.normalize(self.prototypes, dim=-1)
        elif precomputed_prototypes is not None:
            raise ValueError(
                "MPKD.PROTO_ENABLED=False but prototypes were supplied."
            )

    def get_learnable_parameters(self):
        return list(self.student.parameters())

    def get_extra_parameters(self):
        return 0

    def _prototype_probability(self, feature):
        feature = F.normalize(feature.flatten(1), dim=1)
        class_count, center_count, feature_dim = self.prototypes.shape
        centers = self.prototypes.view(class_count * center_count, feature_dim)
        centers = centers.to(dtype=feature.dtype)
        similarities = torch.matmul(feature, centers.t()).view(
            -1, class_count, center_count
        )
        logits = _aggregate_prototype_logits(
            similarities / self.proto_temp, self.proto_aggregation
        )
        return F.softmax(logits, dim=1)

    def _fuse_targets(
        self,
        teacher_probability,
        teacher_probability_raw,
        prototype_probability,
        weak_prediction,
        uncertainty,
    ):
        gate = _prototype_gate(
            teacher_probability,
            prototype_probability,
            weak_prediction,
            uncertainty,
            self.proto_fusion_scale,
            self.proto_fusion_mode,
            self.gate_uncertainty,
            self.gate_agreement,
            self.gate_positive_gain,
            self.gate_confidence,
        ).unsqueeze(1)
        fused = (1.0 - gate) * teacher_probability + gate * prototype_probability
        fused_raw = (
            (1.0 - gate) * teacher_probability_raw
            + gate * prototype_probability
        )
        return _normalize_probability(fused), _normalize_probability(fused_raw)

    def _base_distill_loss(
        self, logits_student, target_probability, target_probability_raw, target, epoch
    ):
        if self.base_method == "KD":
            return self.distill_weight * _kd_from_probability(
                logits_student, target_probability, self.temperature
            )
        if self.base_method == "DKD":
            warmup = min(float(epoch) / self.warmup, 1.0)
            return warmup * _dkd_from_probability(
                logits_student,
                target_probability,
                target,
                self.alpha,
                self.beta,
                self.temperature,
            )
        return self.distill_weight * _nkd_from_probability(
            logits_student,
            target_probability,
            target_probability_raw,
            target,
            self.temperature,
            self.gamma,
        )

    def forward_train(self, image, target, epoch=0, **kwargs):
        image_weak, image_strong = image

        with torch.no_grad():
            logits_t_w, _ = _split_output(self.teacher(image_weak))
            logits_t_s, feature_t_s = _split_output(self.teacher(image_strong))
            probability_t_w_raw = F.softmax(logits_t_w, dim=1)
            probability_t_s_raw = F.softmax(logits_t_s, dim=1)
            probability_t_w = F.softmax(logits_t_w / self.temperature, dim=1)
            probability_t_s = F.softmax(logits_t_s / self.temperature, dim=1)
            # CRLD thresholds confidence on the untempered teacher
            # probabilities. Temperature is only used by the distillation
            # target, so the compatibility adapter keeps that distinction.
            confidence_w = probability_t_w_raw.max(dim=1).values
            confidence_s = probability_t_s_raw.max(dim=1).values
            mask_w = confidence_w.ge(self.tau_w).float()
            mask_s = confidence_s.ge(self.tau_s).float()

            target_t_w = probability_t_w
            target_t_w_raw = probability_t_w_raw
            target_t_s = probability_t_s
            target_t_s_raw = probability_t_s_raw
            if self.proto_enabled:
                if feature_t_s is None:
                    raise RuntimeError(
                        "Teacher output does not contain feats['pooled_feat'] "
                        "required by MPKD."
                    )
                entropy_s = -(
                    probability_t_s_raw
                    * probability_t_s_raw.clamp_min(1e-8).log()
                ).sum(dim=1)
                uncertainty_s = entropy_s / self.max_entropy
                prototype_probability = self._prototype_probability(feature_t_s)
                weak_prediction = probability_t_w_raw.argmax(dim=1)
                target_t_s, target_t_s_raw = self._fuse_targets(
                    probability_t_s,
                    probability_t_s_raw,
                    prototype_probability,
                    weak_prediction,
                    uncertainty_s,
                )

        logits_s_w, _ = _split_output(self.student(image_weak))
        logits_s_s, _ = _split_output(self.student(image_strong))

        loss_ce = self.ce_weight * (
            F.cross_entropy(logits_s_w, target)
            + F.cross_entropy(logits_s_s, target)
        )
        same_weak = self._base_distill_loss(
            logits_s_w, target_t_w, target_t_w_raw, target, epoch
        )
        same_strong = self._base_distill_loss(
            logits_s_s, target_t_s, target_t_s_raw, target, epoch
        )
        loss_wv = self.wv_weight * ((same_weak + same_strong) * mask_w).mean()

        cross_strong = self._base_distill_loss(
            logits_s_s, probability_t_w, probability_t_w_raw, target, epoch
        )
        cross_weak = self._base_distill_loss(
            logits_s_w, probability_t_s, probability_t_s_raw, target, epoch
        )
        loss_cv = self.cv_weight * ((cross_strong + cross_weak) * mask_s).mean()

        return logits_s_w, {
            "loss_ce": loss_ce,
            "loss_kd_wv": loss_wv,
            "loss_kd_cv": loss_cv,
        }

    def forward(self, *args, **kwargs):
        if "image_w" in kwargs and "image_s" in kwargs and "target" in kwargs:
            return self.forward_train(
                (kwargs["image_w"], kwargs["image_s"]),
                kwargs["target"],
                epoch=kwargs.get("epoch", 0),
            )
        if "image" in kwargs:
            logits, _ = _split_output(self.student(kwargs["image"]))
            return logits
        raise ValueError("MPKD requires image_w/image_s/target or image.")
