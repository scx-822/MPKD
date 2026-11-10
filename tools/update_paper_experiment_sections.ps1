param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = 'Stop'
$robustnessFigure = (Resolve-Path 'experiments\robustness\res32x4_res8x4\mpkd_robustness_curves.png').Path

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
    $paragraph.Style = $style
}

function Set-Cell($cell, [string]$value, $style, [bool]$header) {
    $cell.Range.Text = $value
    $cell.Range.Paragraphs.Item(1).Style = $style
    $cell.Range.ParagraphFormat.Alignment = 1
    $cell.VerticalAlignment = 1
    $cell.Range.Font.Name = 'Times New Roman'
    $cell.Range.Font.Size = 9
    $cell.Range.Font.Bold = $header
}

function Insert-TableAtMarker($document, [string]$marker, [object[]]$rows, $cellStyle) {
    $range = Find-TextRange $document $marker
    $markerParagraph = $range.Paragraphs.Item(1).Range
    $insertRange = $document.Range($markerParagraph.End, $markerParagraph.End)
    $table = $document.Tables.Add($insertRange, $rows.Count, $rows[0].Count)
    $table.AllowAutoFit = $true
    $table.Borders.Enable = 1
    for ($rowIndex = 0; $rowIndex -lt $rows.Count; $rowIndex++) {
        for ($columnIndex = 0; $columnIndex -lt $rows[0].Count; $columnIndex++) {
            Set-Cell $table.Cell($rowIndex + 1, $columnIndex + 1) `
                ([string]$rows[$rowIndex][$columnIndex]) $cellStyle ($rowIndex -eq 0)
        }
    }
    $markerRange = Find-TextRange $document $marker
    $markerRange.Text = ''
    return $table
}

function Insert-FigureAtMarker($document, [string]$marker, [string]$path) {
    $range = Find-TextRange $document $marker
    $markerParagraph = $range.Paragraphs.Item(1).Range
    $insertRange = $document.Range($markerParagraph.End, $markerParagraph.End)
    $shape = $document.InlineShapes.AddPicture($path, $false, $true, $insertRange)
    $shape.LockAspectRatio = $true
    $shape.Width = 450
    $shape.Range.ParagraphFormat.Alignment = 1
    return $shape
}

Copy-Item -LiteralPath $Source -Destination $Destination -Force
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $document = $word.Documents.Open($Destination, $false, $false)
    if ($document.Comments.Count -ne 11) {
        throw "Unexpected comment count before edit: comments=$($document.Comments.Count)"
    }
    $baselineComments = $document.Comments.Count
    $baselineRevisions = $document.Revisions.Count
    $already = $document.Content.Duplicate
    $alreadyFind = $already.Find
    $alreadyFind.ClearFormatting()
    $alreadyFind.Text = '(7) Robustness under Strong Disturbances'
    $alreadyFind.Forward = $true
    $alreadyFind.Wrap = 0
    if ($alreadyFind.Execute()) {
        throw 'The robustness section already exists; refusing duplicate insertion.'
    }

    $headingStyle = (Find-TextRange $document '(6) Sensitivity to the Number of Prototypes').Paragraphs.Item(1).Style
    $bodyStyle = (Find-TextRange $document '为评估每类子原型数量').Paragraphs.Item(1).Style
    $captionStyle = (Find-TextRange $document 'Table 10:').Paragraphs.Item(1).Style
    $cellStyle = $document.Tables.Item(10).Cell(2, 1).Range.Paragraphs.Item(1).Style

    $heading7 = '(7) Robustness under Strong Disturbances'
    $intro7 = '为检验方法对强干扰强度变化的稳定性，本文在 CIFAR-100 评估集上对 CRLD 与 MPKD 学生模型进行成对增强评估。具体地，采用相同的索引种子生成 RandAugment 强干扰样本，将操作数量 n 设为 1、2、3、4 和 5，并对每个强度重复三次；该过程仅用于评估，不改变模型参数。表11报告三次评估的均值和标准差，图5给出性能随干扰强度变化的曲线。'
    $caption11 = 'Table 11: Robustness to increasing strong-disturbance intensity on CIFAR-100 (ResNet32×4→ResNet8×4).'
    $analysis7 = '如表11和图5所示，随着 RandAugment 操作数量增加，两个模型的 Top-1 和 Top-5 准确率均下降，说明该评估确实逐步提高了干扰强度。在所有测试强度下，MPKD 的平均准确率均高于 CRLD；当 n=5 时，MPKD 的 Top-1 和 Top-5 分别为 34.64% 和 58.56%，较 CRLD 分别高 0.88 和 0.35 个百分点。该结果表明，原型引导的强干扰目标修正能够在不同扰动强度下保留一定的性能优势，但该实验仅覆盖一个教师—学生组合，不能替代跨架构的鲁棒性统计。'
    $captionFigure5 = 'Figure 5: Robustness of CRLD and MPKD under increasing strong-disturbance intensity.'
    $heading8 = '(8) Deployment Cost Analysis'
    $intro8 = 'MPKD 的原型构建和教师计算仅发生在训练阶段，部署时仅保留学生网络。为验证这一点，本文在 RTX 5060 Laptop GPU 上对相同的 ResNet8×4 学生架构进行推理开销测量，输入批量为 128，并采用相同的预热和重复次数。表12报告参数量、单图 FLOPs、峰值推理显存和单图延迟。'
    $caption12 = 'Table 12: Deployment cost of the unchanged ResNet8×4 student on an RTX 5060 Laptop GPU.'
    $analysis8 = '如表12所示，CRLD 与 MPKD 的学生模型均包含 1.234M 个参数，单图计算量均为 0.3572 GFLOPs，峰值推理显存均为 201.6 MB，测得的单图延迟分别为 0.0952 ms 和 0.0912 ms。两次延迟测量的微小差异属于运行噪声，不能视为方法带来的收益。因此，MPKD 不增加学生模型的推理参数、FLOPs 或额外原型分支；原型构建属于一次性的训练阶段开销。上述延迟和显存数值与硬件及批量设置相关，主要用于说明部署阶段的结构等价性。'

    $block = @(
        $heading7, $intro7, $caption11, '[[TABLE11]]', $analysis7,
        $captionFigure5, '[[FIGURE5]]', $heading8, $intro8, $caption12,
        '[[TABLE12]]', $analysis8
    ) -join ([char]13)
    $conclusion = Find-TextRange $document '5. Conclusion'
    $document.Range($conclusion.Start, $conclusion.Start).InsertBefore($block + [char]13)

    $table11Rows = @(
        @('RandAugment n', 'CRLD Top-1', 'MPKD Top-1', 'CRLD Top-5', 'MPKD Top-5'),
        @('Clean', '77.29', '77.72', '94.52', '94.57'),
        @('1', '63.65±0.08', '64.33±0.11', '86.01±0.17', '86.27±0.10'),
        @('2', '55.08±0.33', '55.37±0.11', '79.30±0.25', '79.82±0.07'),
        @('3', '47.12±0.07', '47.87±0.36', '72.42±0.18', '72.93±0.26'),
        @('4', '39.67±0.10', '40.38±0.22', '64.77±0.26', '65.59±0.26'),
        @('5', '33.76±0.27', '34.64±0.14', '58.21±0.38', '58.56±0.13')
    )
    $table12Rows = @(
        @('Method', 'Params (M)', 'FLOPs/image (G)', 'Peak memory (MB)', 'Latency/image (ms)'),
        @('CRLD student', '1.234', '0.3572', '201.6', '0.0952'),
        @('MPKD student', '1.234', '0.3572', '201.6', '0.0912')
    )
    $table11 = Insert-TableAtMarker $document '[[TABLE11]]' $table11Rows $cellStyle
    for ($columnIndex = 0; $columnIndex -lt $table11Rows[0].Count; $columnIndex++) {
        Set-Cell $table11.Cell(1, $columnIndex + 1) `
            ([string]$table11Rows[0][$columnIndex]) $cellStyle $true
    }
    Insert-FigureAtMarker $document '[[FIGURE5]]' $robustnessFigure | Out-Null
    $table12 = Insert-TableAtMarker $document '[[TABLE12]]' $table12Rows $cellStyle
    for ($columnIndex = 0; $columnIndex -lt $table12Rows[0].Count; $columnIndex++) {
        Set-Cell $table12.Cell(1, $columnIndex + 1) `
            ([string]$table12Rows[0][$columnIndex]) $cellStyle $true
    }

    Set-ParagraphByPrefix $document $heading7 $heading7 $headingStyle
    Set-ParagraphByPrefix $document '为检验方法对强干扰强度变化' $intro7 $bodyStyle
    Set-ParagraphByPrefix $document 'Table 11:' $caption11 $captionStyle
    Set-ParagraphByPrefix $document '如表11和图5所示' $analysis7 $bodyStyle
    Set-ParagraphByPrefix $document 'Figure 5:' $captionFigure5 $captionStyle
    Set-ParagraphByPrefix $document $heading8 $heading8 $headingStyle
    Set-ParagraphByPrefix $document 'MPKD 的原型构建和教师计算' $intro8 $bodyStyle
    Set-ParagraphByPrefix $document 'Table 12:' $caption12 $captionStyle
    Set-ParagraphByPrefix $document '如表12所示' $analysis8 $bodyStyle

    Set-ParagraphByPrefix $document '为评估MPKD在图像分类知识蒸馏任务中的性能' `
        '为评估 MPKD 在图像分类知识蒸馏任务中的性能，本文在 CIFAR-100、Tiny-ImageNet-200 和 ImageNet-1K 上开展实验，并分别报告各数据集相应评估集上的分类结果。' $bodyStyle
    Set-ParagraphByPrefix $document '遵循 CRLD 的实验协议' `
        '遵循 CRLD 的实验协议，本文选用 ResNet、Wide ResNet（WRN）、VGG、MobileNetV2 和 ShuffleNetV2 作为教师与学生骨干网络，以评价 MPKD 在同构和异构蒸馏场景中的表现 [40–44]。' $bodyStyle
    Set-ParagraphByPrefix $document '在 CIFAR-100 上，所有学生模型训练' `
        '在 CIFAR-100 上，所有学生模型训练 240 个 epoch，批量大小为 64，并采用 SGD 优化器。动量设为 0.9，权重衰减设为 5×10−4；常规卷积学生的初始学习率为 0.05，MobileNetV2 与 ShuffleNetV2 学生的初始学习率为 0.01，学习率在第 150、180 和 210 个 epoch 衰减为原来的 0.1 倍。弱干扰视图采用随机裁剪和水平翻转，强干扰视图在此基础上加入 RandAugment [35] 与 Cutout [36]。MPKD 沿用各教师—学生组合对应的 CRLD 蒸馏设置，监督损失权重统一设为 α=1。默认每个类别使用 K=3 个子原型，并通过最多 10 次类内 K-Means 更新完成构建。在线蒸馏掩码阈值 τw 和 τs 与离线原型样本筛选阈值 τproto 分别设置，具体值以各实验 YAML 文件为准。' $bodyStyle
    Set-ParagraphByPrefix $document '在上述方法中，CRLD [3]' `
        '在上述方法中，CRLD [3] 作为本文的主要竞争基线。遵循 CRLD 及相关知识蒸馏工作的评价方式，本文采用 Top-1 Accuracy 和 Top-5 Accuracy 两项指标。Top-1 准确率表示预测得分最高的类别与真实类别一致的评估样本比例；Top-5 准确率表示真实类别位于预测得分最高的五个候选类别中的评估样本比例。' $bodyStyle
    Set-ParagraphByPrefix $document '本文在CIFAR-100上评估MPKD的性能' `
        '本文在 CIFAR-100 上评估 MPKD 的性能。表1和表2分别报告异构与同构教师—学生组合的 Top-1 准确率，其中异构蒸馏表示教师与学生来自不同网络族，同构蒸馏表示二者属于同一网络族。表中括号内数字表示未使用知识蒸馏时相应模型的 Top-1 准确率；对照方法数值取自对应原论文或 CRLD 公开基准，MPKD 数值由本文按所述设置训练得到。' $bodyStyle
    Set-ParagraphByPrefix $document '综合两类架构结果' `
        '综合两类架构结果，MPKD 在多数已测教师—学生组合上优于 CRLD [3]，最高 Top-1 准确率达到 79.38%。该结果说明，原型引导的强干扰教师目标修正能够迁移到不同的同构和异构蒸馏组合，但提升幅度仍取决于具体架构。' $bodyStyle
    Set-ParagraphByPrefix $document '该结果提示，多中心原型的收益可能受到' `
        '该结果提示，多中心原型的收益可能受到数据集规模、输入分辨率、教师特征分布和训练超参数的影响，不能将 CIFAR-100 上的提升直接外推至 Tiny-ImageNet。' $bodyStyle
    Set-ParagraphByPrefix $document '该结果为ResNet56→ResNet20这一设置下的阈值敏感性提供了证据' `
        '该结果为 ResNet56→ResNet20 设置下的阈值敏感性提供了经验观察：适度筛选可能改善原型质量，但单次训练尚不足以验证严格的因果关系。' $bodyStyle

    # The old K=3 value was produced under a different filtering setting. Keep
    # it visible only as pending until the same-protocol K=3 run finishes.
    $kTable = $document.Tables.Item(12)
    Set-Cell $kTable.Cell(4, 2) 'pending' $cellStyle $false
    Set-Cell $kTable.Cell(4, 3) 'pending' $cellStyle $false
    Set-ParagraphByPrefix $document '为评估每类子原型数量' `
        '为评估每类子原型数量 K 对 MPKD 性能的影响，本文在 CIFAR-100 的 ResNet32×4→ResNet8×4 设置下比较 K=1、2、3、4 和 5。K=1、2、4 和 5 已采用相同的 seed、训练轮数、学习率策略、可靠性门控和离线筛选阈值；原有 K=3 数值来自不同的筛选设置，因此在同协议复核完成前不将其与其他四组直接比较。表10中的 pending 表示该行尚待同协议实验确认。' $bodyStyle
    Set-ParagraphByPrefix $document '如表10所示，K=3 在 Top-1' `
        '当前同协议结果中，K=1、2、4 和 5 的 Top-1 分别为 77.23%、77.05%、77.06% 和 77.05%，Top-5 分别为 94.41%、94.78%、94.55% 和 94.46%。四组结果在单次运行下较为接近，尚不能据此判断原型数量的全局最优值；待 K=3 复核完成后，再根据统一协议确定默认设置。' $bodyStyle

    $document.Save()
    if ($document.Comments.Count -ne $baselineComments -or $document.Revisions.Count -ne $baselineRevisions) {
        throw "Annotation count changed: before=$baselineComments/$baselineRevisions, after=$($document.Comments.Count)/$($document.Revisions.Count)"
    }
    Write-Output "Saved: $Destination; paragraphs=$($document.Paragraphs.Count); tables=$($document.Tables.Count); comments=$($document.Comments.Count); revisions=$($document.Revisions.Count)"
    $document.Close([ref]$false)
}
finally {
    $word.Quit()
}




