"""Patch the fourth chapter of the MPKD paper without touching comments.

The document is edited at the OOXML paragraph level so comments, images,
tables, and revision parts remain in the DOCX package unchanged.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import zipfile
from pathlib import Path

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
XML_SPACE = "http://www.w3.org/XML/1998/namespace"


def paragraph_text(paragraph: etree._Element) -> str:
    parts: list[str] = []
    for node in paragraph.xpath(".//w:t", namespaces=NS):
        parts.append(node.text or "")
    return "".join(parts)


def replace_paragraph(paragraph: etree._Element, text: str) -> None:
    ppr = paragraph.find("w:pPr", namespaces=NS)
    first_run = paragraph.find(".//w:r", namespaces=NS)
    run_properties = None
    if first_run is not None:
        old_rpr = first_run.find("w:rPr", namespaces=NS)
        if old_rpr is not None:
            run_properties = copy.deepcopy(old_rpr)

    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)

    run = etree.SubElement(paragraph, f"{{{W_NS}}}r")
    if run_properties is not None:
        run.append(run_properties)
    text_node = etree.SubElement(run, f"{{{W_NS}}}t")
    text_node.set(f"{{{XML_SPACE}}}space", "preserve")
    text_node.text = text


def patch_document(path: Path) -> list[str]:
    with zipfile.ZipFile(path, "r") as archive:
        document_xml = archive.read("word/document.xml")
        root = etree.fromstring(document_xml)

        # These are intentionally limited to body paragraphs. No method
        # equations, table cells, comments, or annotation anchors are edited.
        body_paragraphs = root.xpath("/w:document/w:body/w:p", namespaces=NS)
        replacements = [
            (
                "为评估 MPKD 在图像分类知识蒸馏任务中的性能",
                "为评估 MPKD 在图像分类知识蒸馏任务中的性能，本文在 CIFAR-100、Tiny-ImageNet-200 和 ImageNet-1K 上开展实验，并分别在各数据集既定的评估划分上报告分类结果。",
            ),
            (
                "Tiny-ImageNet-200：",
                "Tiny-ImageNet-200：该数据集包含 200 个类别、100,000 张训练图像和 10,000 张验证图像，图像分辨率为 64×64 [67]。",
            ),
            (
                "ImageNet-1K：",
                "ImageNet-1K：该数据集包含 1,000 个类别，训练集约含 128 万张图像，验证集含 50,000 张图像 [38,68]。",
            ),
            (
                "遵循 CRLD 的实验协议",
                "遵循 CRLD 的实验协议，本文选用 ResNet、Wide ResNet（WRN）、VGG、MobileNetV2 和 ShuffleNetV2 作为教师与学生骨干网络，以评价 MPKD 在同构和异构蒸馏场景中的表现 [40–44]。除待考察模块外，各组尽量沿用相同的教师、学生、数据增强、优化器、训练轮数和随机种子；具体超参数以对应配置文件为准。",
            ),
            (
                "在上述方法中，CRLD [3]",
                "在上述方法中，CRLD [3] 作为本文的主要竞争基线。本文采用 Top-1 Accuracy 和 Top-5 Accuracy 两项指标：Top-1 准确率表示预测得分最高的类别与真实类别一致的评估样本比例，Top-5 准确率表示真实类别位于预测得分最高的五个候选类别中的评估样本比例。表中“−”表示对应文献未报告该指标；公开文献数值与本文复现实验数值来源不同。",
            ),
            (
                "本文在 CIFAR-100 上评估 MPKD 的性能",
                "本文首先在 CIFAR-100 上评估 MPKD 的性能。表1和表2分别报告异构与同构教师—学生组合的 Top-1 准确率，其中异构蒸馏表示教师与学生来自不同网络族，同构蒸馏表示二者属于同一网络族。表中括号内数字表示未使用知识蒸馏时相应模型的 Top-1 准确率；对照方法数值取自对应原论文或 CRLD 公开基准，MPKD 数值由本文按所述配置训练得到。",
            ),
            (
                "如表 1 所示",
                "如表1所示，在异构蒸馏设置中，MPKD 在六组组合中的四组取得最高 Top-1 准确率，在其余两组中取得第二高结果。其平均准确率为 74.76%，较表中 CRLD 结果的 74.10% 提高 0.66 个百分点。该结果支持原型引导的强干扰目标修正能够适应部分网络族差异，但各组合增益并不相同，不能据此断言其对所有异构架构均同等有效。",
            ),
            (
                "如表2所示",
                "如表2所示，在同构蒸馏设置中，MPKD 在六组组合中的五组取得最高 Top-1 准确率，在剩余一组中取得第二高结果。MPKD 的平均准确率为 75.83%，较 CRLD 的 75.24% 提高 0.59 个百分点。六组结果均高于对应的 CRLD 结果，但提升幅度随教师—学生结构而异，说明原型修正的收益仍受到特征分布和训练配置影响。",
            ),
            (
                "综合两类架构结果",
                "综合两类架构结果，MPKD 在当前列出的 CIFAR-100 组合中总体优于 CRLD，最高 Top-1 准确率为 79.38%。这一结果支持原型引导的强干扰教师目标修正具有跨架构迁移能力，但不同组合的提升幅度存在差异，仍需结合消融和跨数据集结果理解。",
            ),
            (
                "该结果提示，多中心原型的收益可能受到",
                "表3给出了 Tiny-ImageNet-200 上的实验结果。CRLD [3] 的 Top-1 和 Top-5 准确率分别为 63.39% 和 84.20%，MPKD 分别为 62.42% 和 84.35%。因此，当前设置下 MPKD 的 Top-1 准确率较 CRLD 低 0.97 个百分点，而 Top-5 准确率高 0.15 个百分点，整体表现并非全面占优。该结果提示，多中心原型的收益可能受到数据集规模、输入分辨率、教师特征分布和训练超参数的影响，不能将 CIFAR-100 上的提升直接外推至 Tiny-ImageNet。",
            ),
            (
                "表4汇总了MPKD",
                "表4汇总了 MPKD 在 ImageNet-1K 上两组教师—学生架构的 Top-1 和 Top-5 准确率。对于 ResNet34→ResNet18，MPKD 分别达到 72.53% 和 90.98%，较 CRLD 提升 0.16 和 0.22 个百分点；对于 ResNet50→MobileNetV1，MPKD 达到 73.61% 和 91.48%，较 CRLD 分别提升 0.08 和 0.05 个百分点。两组设置下，MPKD 的 Top-1 均高于表中其他对照方法；Top-5 则在 ResNet34→ResNet18 上取得最高结果，而在 ResNet50→MobileNetV1 上低于 TopKD。总体而言，MPKD 在这两组 ImageNet-1K 设置中相较 CRLD 获得小幅提升，但该结果不应外推为对所有架构组合的普遍优势。",
            ),
            (
                "为验证MPKD是否依赖",
                "为验证 MPKD 是否依赖某一种特定的 Logit 蒸馏损失，表5分别在 KD [1]、DKD [24] 和 NKD [25] 三种基础方法上进行兼容性实验，并与相应的“基础方法+CRLD [3]”组合比较。六组教师—学生架构的平均结果显示，KD+MPKD、DKD+MPKD 和 NKD+MPKD 分别达到 75.83%、75.28% 和 75.81%，较对应的 KD+CRLD、DKD+CRLD 和 NKD+CRLD 分别提高 0.59、0.44 和 0.62 个百分点。NKD+MPKD 在六组架构上均优于 NKD+CRLD；DKD+MPKD 在个别架构上略有波动，说明兼容性收益具有架构依赖性。该实验支持 MPKD 可作为训练阶段目标修正模块接入多种 Logit 蒸馏目标，但并不构成多随机种子统计检验。",
            ),
            (
                "为进一步分析不同组成模块",
                "为进一步分析不同组成模块对模型性能的影响，本文在相同教师、学生、数据增强、优化器和训练轮数设置下进行视图构建、原型目标修正和可靠性门控消融。CIFAR-100 上 ResNet110→ResNet32 的结果如表6所示。",
            ),
            (
                "为直观分析多中心原型",
                "为直观分析多中心原型所保留的类内结构，本文在 CIFAR-100 的 ResNet32×4 教师模型上，按照原型构建阶段的可靠样本筛选规则提取弱干扰特征，并在 K=3 设置下为每类构建三个子原型。图2选取 8 个类别作为示例，展示每类 120 个可靠教师特征及对应子原型在 t-SNE [39] 二维空间中的分布，星形标记表示子原型中心。特征呈现类内局部差异，多个子原型分布于不同局部区域；该图为多中心建模提供定性支持，但 t-SNE 仅用于可视化，不能替代定量比较。",
            ),
            (
                "为直接验证原型修正部分",
                "为直接验证原型修正部分的作用，本文采用 CIFAR-100 ResNet110→ResNet32 消融结果进行逐步比较。配置 III 为不含原型修正的完整 CRLD [3] 双视图基线，Top-1 为 74.42%；配置 IV 加入原型类别分布对强干扰教师目标的修正后达到 74.99%；配置 V 再加入可靠性门控后达到 75.21%。性能增益首先出现在原型目标修正之后，门控则进一步限制不可靠结构证据的传播。该结果支持 MPKD 的主要作用来自强干扰目标重构，而非额外引入独立特征损失。",
            ),
            (
                "该结果为 ResNet56→ResNet20",
                "表7和图4报告了 CIFAR-100 上 ResNet56→ResNet20 设置的原型样本保留率及学生 Top-1 准确率。τproto=0 时关闭置信度门槛，但仍保留教师弱视图预测正确条件，因此 97.786% 的保留率反映正确性筛选后的覆盖率。τproto=0.6 时保留率为 97.244%，Top-1 达到 72.91%，较 τproto=0 提高 0.62 个百分点；τproto=0.8 和 0.95 时保留率进一步降至 92.358% 和 69.498%，准确率分别为 72.41% 和 72.64%，并未随筛选增强而单调提升。τproto=0 与 0.2 的保留率相同而 Top-1 略有差异，也说明单次训练的精度变化不能仅由保留率解释。该表显示适度置信度筛选在此设置下有潜在收益，但单架构、单随机种子结果不足以确定全局最优阈值。",
            ),
            (
                "为比较类内多中心原型的聚合方式",
                "为比较类内多中心原型的聚合方式，本文在 CIFAR-100 的 ResNet32×4→ResNet8×4 设置下评估 Log-Mean-Exp、Mean 和 Max。三组实验采用相同训练协议、随机种子、原型数量、筛选阈值和可靠性门控，仅改变聚合函数。Log-Mean-Exp 的 Top-1/Top-5 为 77.72%/95.27%，较 Mean 分别提高 0.39/0.54 个百分点，较 Max 分别提高 0.72/0.83 个百分点，说明在当前设置中，平滑整合多个局部原型响应优于简单平均或仅取最大响应。",
            ),
            (
                "如表 9 所示",
                "如表9所示，采用弱视图教师预测类别作为锚点时，最高 Top-1/Top-5 准确率为 72.91%/93.18%；改用真实标签后为 72.33%/93.14%，分别下降 0.58 和 0.04 个百分点。真实标签仍用于交叉熵监督及离线可靠样本筛选；该消融仅改变在线目标重构的语义锚点。结果支持在当前设置下使用弱视图教师预测作为视图内参照，但该单架构、单随机种子对比应视为实验观察，而非普遍结论。",
            ),
            (
                "如表11和图5所示",
                "如表11和图5所示，随着 RandAugment 操作数量增加，CRLD 与 MPKD 的 Top-1 和 Top-5 均下降，说明评估中的强干扰强度逐步提高。在所有测试强度下，MPKD 的均值均高于 CRLD；当 n=5 时，MPKD 的 Top-1 和 Top-5 分别为 34.64% 和 58.56%，较 CRLD 高 0.88 和 0.35 个百分点。该结果表明 MPKD 在当前教师—学生组合上对强干扰具有一定稳定性，但不能替代跨架构鲁棒性统计。",
            ),
            (
                "MPKD 的原型构建和教师计算",
                "MPKD 的原型构建和教师计算仅发生在训练阶段，部署时仅保留学生网络。本文在 RTX 5060 Laptop GPU 上对结构完全相同的 ResNet8×4 学生架构进行推理开销测量，输入批量为 128，并采用相同预热和重复次数。表12报告参数量、单图 FLOPs、峰值推理显存和单图延迟。",
            ),
            (
                "如表12所示",
                "如表12所示，CRLD 与 MPKD 的学生模型均包含 1.234M 个参数，单图计算量均为 0.3572 GFLOPs，峰值推理显存均为 201.6 MB。测得的单图延迟分别为 0.0952 ms 和 0.0912 ms，微小差异属于测量噪声，不能视为 MPKD 带来的速度提升。因此，MPKD 不增加学生模型的推理参数、FLOPs 或额外原型分支；原型构建属于训练阶段的一次性开销。该测量值受硬件、批量和测量设置影响，主要用于说明部署结构等价性。",
            ),
            (
                "本文提出多中心原型知识蒸馏方法MPKD",
                "本文提出多中心原型知识蒸馏方法 MPKD，旨在缓解强干扰条件下教师预测偏移造成的不可靠监督传播。MPKD 利用教师弱干扰特征构建类条件多中心原型，并依据强干扰特征与各类别原型的匹配关系，通过可靠性门控按需修正教师目标。CIFAR-100 主实验与消融显示，原型目标重构及门控在多个教师—学生组合上带来性能增益；ImageNet-1K 两组结果相较 CRLD 有小幅提升，而 Tiny-ImageNet-200 上的 Top-1 尚未优于 CRLD，说明收益仍与数据集和架构有关。鲁棒性评估显示，MPKD 在当前组合的不同强干扰强度下保持小幅优势；原型数量、聚合方式和语义锚点消融进一步刻画了方法的适用范围。由于原型仅用于训练阶段的目标修正，部署时只保留学生网络，不增加推理参数、FLOPs 或原型分支。",
            ),
            ("Table 3：", "Table 3: Top-1 and Top-5 accuracy (%) on Tiny-ImageNet."),
            ("Table 6: Ablation study", "Table 6: Ablation study on view construction and target reconstruction on CIFAR-100 (ResNet110→ResNet32)."),
            ("4.4.3 Prototype Structure and Target Reconstruction Analysis", "4.4.3 Prototype, Robustness, and Deployment Analysis"),
        ]

        changed: list[str] = []
        for prefix, replacement in replacements:
            matches = [p for p in body_paragraphs if paragraph_text(p).startswith(prefix)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one paragraph for {prefix!r}, found {len(matches)}")
            replace_paragraph(matches[0], replacement)
            changed.append(prefix)

        patched_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        temp = path.with_suffix(".patched.docx")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for item in archive.infolist():
                data = patched_xml if item.filename == "word/document.xml" else archive.read(item.filename)
                out.writestr(item, data)
    shutil.move(str(temp), str(path))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    args = parser.parse_args()
    backup = args.document.with_name(args.document.stem + ".before_fourth_chapter_patch.docx")
    shutil.copy2(args.document, backup)
    changed = patch_document(args.document)
    print(f"Patched {args.document}")
    print(f"Backup {backup}")
    print(f"Paragraphs changed: {len(changed)}")


if __name__ == "__main__":
    main()
