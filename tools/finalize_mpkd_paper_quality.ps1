param(
    [Parameter(Mandatory = $true)]
    [string]$DocumentPath
)

$ErrorActionPreference = 'Stop'

function Find-TextRange($document, [string]$text) {
    $range = $document.Content.Duplicate()
    $find = $range.Find
    $find.ClearFormatting()
    $find.Text = $text
    $find.Forward = $true
    $find.Wrap = 0
    if (-not $find.Execute()) {
        throw "Text not found: $text"
    }
    return $range
}

function Set-ParagraphByPrefix($document, [string]$prefix, [string]$text, $style) {
    $found = Find-TextRange $document $prefix
    $paragraph = $found.Paragraphs.Item(1).Range
    $paragraph.Text = $text + [char]13
    if ($null -ne $style) {
        $paragraph.Style = $style
    }
}

function Set-Cell($cell, [string]$value, $style, [bool]$header = $false) {
    $cell.Range.Text = $value
    if ($null -ne $style) {
        $cell.Range.Paragraphs.Item(1).Style = $style
    }
    $cell.Range.ParagraphFormat.Alignment = 1
    $cell.VerticalAlignment = 1
    $cell.Range.Font.Name = 'Times New Roman'
    $cell.Range.Font.Size = 9
    $cell.Range.Font.Bold = $header
}

function Get-TableByFirstCell($document, [string]$value) {
    for ($i = 1; $i -le $document.Tables.Count; $i++) {
        $table = $document.Tables.Item($i)
        $text = ($table.Cell(1, 1).Range.Text -replace '[\r\a]', '').Trim()
        if ($text -eq $value) {
            return $table
        }
    }
    throw "Table with first cell '$value' not found."
}

function Get-TableByRowLabel($document, [string]$rowLabel) {
    for ($i = 1; $i -le $document.Tables.Count; $i++) {
        $table = $document.Tables.Item($i)
        for ($r = 1; $r -le $table.Rows.Count; $r++) {
            $text = ($table.Cell($r, 1).Range.Text -replace '[\r\a]', '').Trim()
            if ($text -eq $rowLabel) {
                return $table
            }
        }
    }
    throw "Table containing row '$rowLabel' not found."
}

function Set-RowByFirstCell($table, [string]$rowLabel, [string[]]$values, $style) {
    for ($r = 1; $r -le $table.Rows.Count; $r++) {
        $label = ($table.Cell($r, 1).Range.Text -replace '[\r\a]', '').Trim()
        if ($label -eq $rowLabel) {
            for ($c = 0; $c -lt $values.Count; $c++) {
                Set-Cell $table.Cell($r, $c + 1) $values[$c] $style ($r -eq 1)
            }
            return
        }
    }
    throw "Row '$rowLabel' not found."
}

$backup = Join-Path $env:TEMP ("MPKD_paper_before_quality_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.docx')
Copy-Item -LiteralPath $DocumentPath -Destination $backup -Force

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $document = $word.Documents.Open($DocumentPath, $false, $false)
    $baselineComments = $document.Comments.Count
    $baselineRevisions = $document.Revisions.Count
    if ($baselineComments -ne 11) {
        throw "Unexpected comment count: $baselineComments"
    }

    $headingStyle = (Find-TextRange $document '(6) Sensitivity to the Number of Prototypes').Paragraphs.Item(1).Style
    $bodyStyle = (Find-TextRange $document '为评估每类子原型数量').Paragraphs.Item(1).Style
    $captionStyle = (Find-TextRange $document 'Table 10:').Paragraphs.Item(1).Style
    $cellStyle = (Get-TableByFirstCell $document 'K').Cell(2, 1).Range.Paragraphs.Item(1).Style

    $replacements = @(
        @('知识蒸馏通过传递教师模型输出中的软类别关系', '知识蒸馏通过传递教师模型输出中的软类别关系，为轻量学生模型提供比硬标签更丰富的监督信息。然而，当输入受到不同程度的干扰时，教师预测的可靠性可能发生变化；尤其在强干扰条件下，类别偏移或置信度失配可能将不可靠信息传递给学生。为此，本文提出多中心原型知识蒸馏（Multi-center Prototype Knowledge Distillation，MPKD）。该方法利用教师模型在弱干扰条件下提取的可靠特征构建类条件多中心原型，以刻画类别内部的局部多模态结构；随后依据强干扰特征与各类别原型的匹配关系，获得独立于分类头输出的类别结构证据。MPKD以弱干扰条件下的教师预测类别为语义锚点，通过可靠性门控按需修正强干扰条件下的教师目标，并自适应调节原型证据的融合强度。当结构证据不足或修正方向不一致时，方法保留原始教师目标。实验结果表明，MPKD在CIFAR-100的多组架构和ImageNet-1K的两组架构上取得了有竞争力的结果；在Tiny-ImageNet-200上，方法的Top-1结果尚未全面优于对照方法，说明其收益仍受数据集和架构影响。由于原型仅用于训练阶段的目标修正，推理阶段不会增加学生模型的参数量或计算量。'),
        @('关键词：', '关键词：知识蒸馏；多中心原型；强干扰教师目标修正；可靠性门控'),
        @('随着深度神经网络规模和计算复杂度不断增加', '随着深度神经网络的规模和计算复杂度不断增加，如何在资源受限的边缘设备上部署兼具效率与可靠性的轻量模型，已成为模型压缩和视觉识别领域的重要问题。知识蒸馏通过将高容量教师模型中的判别知识迁移至轻量学生模型，为降低部署成本提供了有效途径。经典知识蒸馏通常将教师模型的logits转换为软目标，使学生学习完整类别空间中的概率关系。与仅依赖one-hot标签的监督学习相比，软目标能够保留类别相似性和决策边界等细粒度信息，因此已成为轻量视觉模型训练的基础方法[1,45–51]。'),
        @('为了进一步提高学生模型在复杂环境中的泛化能力', '为了进一步提高学生模型在复杂环境中的泛化能力，近年来的研究将增强一致性约束引入训练过程。这类方法为同一样本构造不同程度的输入扰动，并要求模型在扰动前后保持相对稳定的类别判断，从而提升模型对外观变化、几何变化和局部遮挡的适应能力[2,3,52–55]。弱干扰通常保留较完整的类别语义，可为目标判定提供相对稳定的参照；强干扰则模拟更具挑战性的输入变化，并提供鲁棒性训练信号[56–59]。然而，强干扰也会增加教师模型发生类别混淆和置信度失配的风险，使类别知识与干扰诱导的预测偏差同时出现在教师目标中。'),
        @('针对干扰条件下的教师预测偏差', '针对干扰条件下的教师预测偏差，现有方法通常依据置信度、预测熵或其他不确定性度量筛选蒸馏样本。跨视图一致性蒸馏通过软标签选择机制舍弃低置信度的受扰动样本，以减少不可靠监督的影响；其他工作也利用校准、不确定性估计和分布偏移评估分析预测可靠性[3,4,60–63]。这些策略能够抑制部分噪声，但主要在样本层面决定是否使用教师目标，难以判断被保留软分布中的哪些类别关系来自稳定语义、哪些成分由输入干扰引起。过于宽松的全分布对齐可能传递教师偏差，而过于严格的硬阈值又可能丢失低置信样本中仍有价值的非目标类暗知识。'),
        @('从特征空间角度看', '从特征空间角度看，受干扰状态下的教师预测可以被视为稳定类别结构与干扰相关变化共同作用的结果。稳定类别结构反映样本的核心语义，是学生模型应当继承的主要知识；干扰相关变化可能表现为类别边界附近的偏移，其中也可能包含有助于提升环境适应性的局部信息。若缺乏显式的类别结构参照，直接拟合完整软分布会将偏差一并传递，直接丢弃低质量样本又会造成细粒度知识损失。因此，需要一种能够从可靠样本中建立类别结构，并对受干扰教师目标进行连续、保守修正的蒸馏机制。'),
        @('为此，本文提出多中心原型知识蒸馏', '为此，本文提出多中心原型知识蒸馏（MPKD）。MPKD首先从教师模型在弱干扰条件下提取的高可靠特征中构建类条件多中心原型，以描述类别内部可能存在的多个局部模式；随后利用教师模型在强干扰条件下的特征与各类原型之间的匹配关系，形成独立于分类头输出的类别结构证据。该证据以弱干扰条件下的预测类别作为语义参照，并由修正需求、类别一致性、正向概率增益和原型置信度共同决定是否进行强干扰教师目标修正及其幅度。当原型证据不足或修正方向不可靠时，方法回退到原始教师目标，从而避免将不确定的结构信息强行注入学生训练。'),
        @('与直接增加学生特征约束或重新设计完整蒸馏损失的做法不同', '与直接增加学生特征约束或重新设计完整蒸馏损失的做法不同，MPKD将多中心原型限定为训练阶段的类别结构证据，仅用于修正强干扰条件下的教师目标，其他监督与蒸馏路径保持原有定义。原型构建在训练前离线完成，并在学生优化过程中保持固定；因此，推理阶段只需使用训练完成的学生模型，不引入额外参数或计算分支。'),
        @('提出MPKD，一种面向强干扰教师预测偏差', '提出MPKD，一种面向强干扰教师预测偏差的多中心原型知识蒸馏方法，从可靠弱干扰特征中提取类别结构证据，用于辅助受干扰教师目标的选择性修正。'),
        @('构建类条件多中心原型，进一步设计可靠性门控机制', '构建类条件多中心原型，并设计可靠性门控机制，联合评估目标修正需求、类别方向一致性、正向概率增益和原型置信度，以连续方式控制结构证据的引入强度；在证据不足时保留原始教师目标。'),
        @('在多种同构和异构教师–学生架构上进行实验', '在多种同构和异构教师—学生架构上进行实验，并结合原型数量、聚合方式、语义锚点、强干扰强度和部署开销分析，系统考察MPKD的适用范围。'),
        @('知识蒸馏通过教师模型的软输出传递类别间关系', '知识蒸馏通过教师模型的软输出传递类别间关系。经典KD使用温度参数平滑教师分布，并以蒸馏距离约束学生模型；后续研究从目标类与非目标类解耦、logit归一化和温度调节等方面改进分布对齐[1,24,25,27–30,45–51]。此类方法主要在输出空间处理教师信号。当输入受到强干扰时，教师分布可能同时包含稳定语义和干扰偏差；仅调整温度或蒸馏距离难以识别错误的类别方向。MPKD在分布蒸馏基础上引入特征空间中的类条件结构证据，对受干扰教师目标进行有限修正。'),
        @('一致性学习要求同一样本在不同干扰条件下保持相对稳定的预测', '一致性学习要求同一样本在不同干扰条件下保持相对稳定的预测，从而提升模型对输入变化的适应能力。FixMatch等方法利用弱干扰预测约束强干扰输入；跨视图一致性知识蒸馏进一步将这一思想用于教师—学生训练，通过同条件和跨条件蒸馏传递共享类别信息[2,3,52–55]。这类方法扩大了监督信号的覆盖范围，但通常依赖置信度筛选决定教师目标是否保留，难以修正被保留分布中的类别偏差；过于严格的阈值又可能丢失低置信目标中的细粒度知识。MPKD保留多干扰蒸馏的训练形式，仅利用类别结构证据对强干扰条件下的教师目标进行条件性修正。'),
        @('原型学习利用具有代表性的中心描述类别结构', '原型学习利用具有代表性的中心描述类别结构，并通过样本与中心之间的相似度完成分类或度量。Prototypical Networks和相关目标结构学习工作表明，原型能够从特征空间补充分类器输出中的类别结构信息[7,8,65,66]。针对单一中心难以表达类内差异的问题，SoftTriple使用多个类别中心刻画局部模式[5]；在归一化特征空间中，球面K-Means可与von Mises–Fisher混合模型建立联系[6,64]。MPKD借鉴多中心表示思想，但不将原型用于学生特征的反向传播，而是将其转换为类别概率分布，用于评估并修正受干扰教师目标。'),
        @('综上，现有研究尚未充分利用类内多模态原型', '综上，现有研究较少将类内多模态原型用于受干扰教师目标的可靠修正。本文针对这一问题提出MPKD，并在输出空间一致性与特征空间类别结构之间建立受门控的连接。'),
        @('为评估 MPKD 在图像分类知识蒸馏任务中的性能', '为评估MPKD在图像分类知识蒸馏任务中的性能，本文在CIFAR-100、Tiny-ImageNet-200和ImageNet-1K上开展实验，并分别在CIFAR-100测试集、Tiny-ImageNet-200验证集和ImageNet-1K验证集上报告分类结果。'),
        @('遵循 CRLD 的实验协议', '遵循CRLD的实验协议，本文选用ResNet、Wide ResNet（WRN）、VGG、MobileNetV2和ShuffleNetV2作为教师与学生骨干网络，以评价MPKD在同构和异构蒸馏场景中的表现[40–44]。除明确考察的因素外，各组实验保持教师、学生、数据增强、优化器、训练轮数和随机种子一致，具体超参数以对应配置文件为准。'),
        @('在 CIFAR-100 上，所有学生模型训练', '在CIFAR-100上，所有学生模型训练240个epoch，批量大小为64，并采用SGD优化器。动量设为0.9，权重衰减设为5×10−4；常规卷积学生的初始学习率为0.05，MobileNetV2与ShuffleNetV2学生的初始学习率为0.01，学习率在第150、180和210个epoch衰减为原来的0.1倍。弱干扰视图采用随机裁剪和水平翻转，强干扰视图在此基础上加入RandAugment[35]与Cutout[36]。MPKD沿用各教师—学生组合对应的CRLD蒸馏设置，监督损失权重统一设为α=1。默认每个类别使用K=3个子原型，并通过最多10次类内K-Means更新完成构建。在线蒸馏掩码阈值τw和τs与离线原型样本筛选阈值τproto分别设置，具体值以各实验YAML文件为准。'),
        @('为全面评价 MPKD 的性能', '为全面评价MPKD的性能，本文参考CRLD及相关知识蒸馏工作的实验设置，选取26种具有代表性的基线方法进行比较。根据主要知识传递对象和技术路线，基线方法划分为六类：'),
        @('在上述方法中，CRLD [3] 作为本文的主要竞争基线', '在上述方法中，CRLD[3]作为本文的主要竞争基线。本文采用Top-1 Accuracy和Top-5 Accuracy两项指标：Top-1准确率表示预测得分最高的类别与真实类别一致的评估样本比例；Top-5准确率表示真实类别位于预测得分最高的五个候选类别中的评估样本比例。表中“−”表示对应文献未报告该指标；公开文献数值与本文复现实验数值来源不同。'),
        @('表3给出了 Tiny-ImageNet-200 上的实验结果', '表3给出了Tiny-ImageNet-200上的实验结果。CRLD[3]的Top-1和Top-5准确率分别为63.39%和84.20%，MPKD分别为62.42%和84.35%。因此，在当前设置下MPKD的Top-1准确率较CRLD低0.97个百分点，而Top-5准确率高0.15个百分点，整体表现并非全面占优。该结果说明，多中心原型的收益可能受到数据集规模、输入分辨率、教师特征分布和训练超参数影响，不能将CIFAR-100上的提升直接外推至Tiny-ImageNet。'),
        @('表4汇总了 MPKD 在 ImageNet-1K 上', '表4汇总了MPKD在ImageNet-1K上两组教师—学生架构的Top-1和Top-5准确率。对于ResNet34→ResNet18，MPKD分别达到72.53%和90.98%，较CRLD提升0.16和0.22个百分点；对于ResNet50→MobileNetV1，MPKD达到73.61%和91.48%，较CRLD分别提升0.08和0.05个百分点。两组设置下，MPKD的Top-1均高于表中其他对照方法；Top-5则在ResNet34→ResNet18上取得最高结果，而在ResNet50→MobileNetV1上低于TopKD。总体而言，MPKD在这两组ImageNet-1K设置中相较CRLD获得小幅提升，但该结果不应外推为对所有架构组合的普遍优势。')
    )

    foreach ($change in $replacements) {
        Set-ParagraphByPrefix $document $change[0] $change[1] $bodyStyle
    }

    Set-ParagraphByPrefix $document '4.4.3 Prototype, Robustness, and Deployment Analysis' '4.4.3 Prototype, Robustness, and Deployment Analysis' $headingStyle
    Set-ParagraphByPrefix $document '为评估每类子原型数量 K' '为评估每类子原型数量K对MPKD性能的影响，本文在CIFAR-100的ResNet32×4→ResNet8×4设置下比较K=1、2、3、4和5。除K外，五组实验采用相同的训练配置、随机种子、学习率策略、原型筛选阈值、聚合方式和可靠性门控；K=3使用同协议复核结果。表10报告训练过程中按各指标选出的最高Top-1和Top-5准确率，两项最高值可能对应不同epoch。' $bodyStyle
    Set-ParagraphByPrefix $document '当前同协议结果中，K=1、2、4 和 5' '如表10所示，K=3取得最高Top-1准确率77.35%，而K=2取得最高Top-5准确率94.78%；K=1、K=4和K=5的Top-1分别为77.23%、77.06%和77.05%，K=3的Top-5为94.59%。因此，K=3在主要指标Top-1上表现最佳，同时K=1和K=2的结果也较为接近，说明原型数量的收益并非简单随K增加而单调变化。综合Top-1表现、表示能力和计算开销，本文采用K=3作为默认设置。该结论来自单一架构和随机种子，主要用于当前实验中的参数选择，不能据此断言K=3对所有数据集和架构均为全局最优。' $bodyStyle
    Set-ParagraphByPrefix $document '为检验方法对强干扰强度变化' '为检验方法对强干扰强度变化的稳定性，本文在CIFAR-100评估集上对CRLD与MPKD学生模型进行成对增强评估。具体地，采用相同的索引种子生成RandAugment强干扰样本，将操作数量n设为1、2、3、4和5，并对每个强度重复三次；该过程仅用于评估，不改变模型参数。表11报告三次评估的均值和标准差，图5给出性能随干扰强度变化的曲线。' $bodyStyle
    Set-ParagraphByPrefix $document 'Table 11: Robustness to increasing' 'Table 11: Robustness to increasing strong-disturbance intensity on CIFAR-100 (ResNet32×4→ResNet8×4; mean±standard deviation over three repeats).' $captionStyle
    Set-ParagraphByPrefix $document '如表11和图5所示' '如表11和图5所示，随着RandAugment操作数量增加，CRLD和MPKD的Top-1与Top-5准确率均下降，说明评估中的强干扰强度逐步提高。MPKD在n=1和n=4时的部分指标略高于CRLD，但在n=2、3和5时并未稳定领先，且差异大多与三次评估的标准差处于同一量级。因此，在当前单一教师—学生组合和单次训练检查点下，MPKD表现出与CRLD相近的强干扰退化趋势，但尚不能据此宣称具有普遍的鲁棒性优势。' $bodyStyle
    Set-ParagraphByPrefix $document 'MPKD 的原型构建和教师计算仅发生' 'MPKD的原型构建和教师计算仅发生在训练阶段，部署时仅保留学生网络。本文在RTX 5060 Laptop GPU上对结构完全相同的ResNet8×4学生架构进行推理开销测量，输入批量为128，预热50次并重复测量200次。表12报告参数量、单图FLOPs、峰值推理显存和单图延迟。' $bodyStyle
    Set-ParagraphByPrefix $document 'Table 12: Deployment cost' 'Table 12: Deployment cost of the unchanged ResNet8×4 student on an RTX 5060 Laptop GPU (batch size 128).' $captionStyle
    Set-ParagraphByPrefix $document '如表12所示' '如表12所示，CRLD与MPKD的学生模型均包含1.234M个参数，单图计算量均为0.3572 GFLOPs，峰值推理显存均为201.6MB；实测单图延迟分别为0.0465ms和0.0472ms。两次延迟测量的微小差异属于运行噪声，不能视为MPKD带来的速度变化。因此，MPKD不增加学生模型的推理参数、FLOPs或额外原型分支；原型构建属于一次性的训练阶段开销。上述延迟和显存数值受硬件、批量和测量设置影响，主要用于说明部署阶段的结构等价性。' $bodyStyle
    Set-ParagraphByPrefix $document '本文提出多中心原型知识蒸馏方法 MPKD' '本文提出多中心原型知识蒸馏方法MPKD，旨在缓解强干扰条件下教师预测偏移造成的不可靠监督传播。MPKD利用教师弱干扰特征构建类条件多中心原型，并依据强干扰特征与各类别原型的匹配关系，通过可靠性门控按需修正教师目标。CIFAR-100主实验和消融实验表明，原型目标重构及门控在多组教师—学生组合中具有积极作用；ImageNet-1K两组结果相较CRLD有小幅提升，而Tiny-ImageNet-200上的Top-1尚未优于CRLD，说明收益仍与数据集和架构有关。公平的K消融表明，K=3在主要Top-1指标上取得最佳结果；强干扰评估则显示，MPKD在当前设置下保持与CRLD相近的性能退化趋势。由于原型仅用于训练阶段的目标修正，部署时只保留学生网络，不增加推理参数、FLOPs或原型分支。' $bodyStyle

    $kTable = Get-TableByFirstCell $document 'K'
    Set-RowByFirstCell $kTable '3' @('3', '77.35', '94.59') $cellStyle

    $robustTable = Get-TableByFirstCell $document 'RandAugment n'
    $robustRows = @{
        'Clean' = @('Clean', '77.29', '77.35', '94.52', '94.53')
        '1' = @('1', '63.65±0.08', '63.59±0.42', '86.01±0.17', '86.19±0.24')
        '2' = @('2', '55.08±0.33', '54.95±0.29', '79.30±0.25', '79.29±0.33')
        '3' = @('3', '47.12±0.07', '47.10±0.18', '72.42±0.18', '72.13±0.23')
        '4' = @('4', '39.67±0.10', '39.73±0.38', '64.77±0.26', '64.98±0.23')
        '5' = @('5', '33.76±0.27', '33.62±0.27', '58.21±0.38', '58.13±0.10')
    }
    foreach ($key in $robustRows.Keys) {
        Set-RowByFirstCell $robustTable $key $robustRows[$key] $cellStyle
    }

    $overheadTable = Get-TableByRowLabel $document 'CRLD student'
    Set-RowByFirstCell $overheadTable 'CRLD student' @('CRLD student', '1.234', '0.3572', '201.6', '0.0465') $cellStyle
    Set-RowByFirstCell $overheadTable 'MPKD student' @('MPKD student', '1.234', '0.3572', '201.6', '0.0472') $cellStyle

    $document.Save()
    if ($document.Comments.Count -ne $baselineComments -or $document.Revisions.Count -ne $baselineRevisions) {
        throw "Annotation count changed: before=$baselineComments/$baselineRevisions, after=$($document.Comments.Count)/$($document.Revisions.Count)"
    }
    Write-Output "Saved: $DocumentPath"
    Write-Output "Backup: $backup"
    Write-Output "Paragraphs=$($document.Paragraphs.Count); Tables=$($document.Tables.Count); Comments=$($document.Comments.Count); Revisions=$($document.Revisions.Count)"
    $document.Close([ref]$false)
}
finally {
    $word.Quit()
}
