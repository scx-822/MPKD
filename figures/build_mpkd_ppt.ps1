$ErrorActionPreference = 'Stop'

$target = 'C:\Users\scx\Desktop\pig.pptx'
$source = Join-Path $PSScriptRoot 'assets\mpkd_cat_source.jpg'
if (-not (Test-Path -LiteralPath $target)) { throw "Missing target: $target" }
if (-not (Test-Path -LiteralPath $source)) { throw "Missing cat image: $source" }

Add-Type -AssemblyName System.Drawing
$photo = [Drawing.Bitmap]::FromFile($source)
$views = @()
try {
    $specs = @(
        @{ crop = [Drawing.Rectangle]::new(100, 0, 440, 440); flip = $false; tint = $false },
        @{ crop = [Drawing.Rectangle]::new(170, 24, 235, 235); flip = $true;  tint = $false },
        @{ crop = [Drawing.Rectangle]::new(190, 0, 250, 250); flip = $false; tint = $true }
    )
    foreach ($spec in $specs) {
        $out = [Drawing.Bitmap]::new(320, 320)
        $g = [Drawing.Graphics]::FromImage($out)
        try {
            $g.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $c = $spec.crop
            $dst = [Drawing.Rectangle]::new(0, 0, 320, 320)
            if ($spec.tint) {
                $attr = [Drawing.Imaging.ImageAttributes]::new()
                try {
                    $matrix = [Drawing.Imaging.ColorMatrix]::new()
                    $matrix.Matrix00 = 1.48
                    $matrix.Matrix11 = 1.10
                    $matrix.Matrix22 = 0.62
                    $matrix.Matrix40 = -0.04
                    $matrix.Matrix41 = -0.02
                    $matrix.Matrix42 = 0.04
                    $attr.SetColorMatrix($matrix)
                    $g.DrawImage($photo, $dst, $c.X, $c.Y, $c.Width, $c.Height, [Drawing.GraphicsUnit]::Pixel, $attr)
                } finally { $attr.Dispose() }
                $cutout = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(132, 138, 141))
                try { $g.FillRectangle($cutout, 175, 163, 112, 108) }
                finally { $cutout.Dispose() }
            } else {
                $g.DrawImage($photo, $dst, $c, [Drawing.GraphicsUnit]::Pixel)
            }
        } finally { $g.Dispose() }
        if ($spec.flip) { $out.RotateFlip([Drawing.RotateFlipType]::RotateNoneFlipX) }
        $path = Join-Path $env:TEMP ("mpkd_ppt_cat_$([guid]::NewGuid().ToString('N')).png")
        $out.Save($path, [Drawing.Imaging.ImageFormat]::Png)
        $out.Dispose()
        $views += $path
    }
} finally { $photo.Dispose() }

function color([string]$hex) {
    $v = $hex.TrimStart('#')
    [int]::Parse($v.Substring(0,2), 'HexNumber') -bor
    ([int]::Parse($v.Substring(2,2), 'HexNumber') -shl 8) -bor
    ([int]::Parse($v.Substring(4,2), 'HexNumber') -shl 16)
}
function box([string]$name,[double]$x,[double]$y,[double]$w,[double]$h,[string]$fill,[string]$stroke,[double]$weight=1.5,[bool]$dashed=$false) {
    $s = $slide.Shapes.AddShape(1,$x,$y,$w,$h)
    $s.Name = "MPKD_$name"
    $s.Fill.ForeColor.RGB = color $fill
    $s.Line.ForeColor.RGB = color $stroke
    $s.Line.Weight = $weight
    if ($dashed) { $s.Line.DashStyle = 4 }
    $s
}
function label([string]$name,[string]$text,[double]$x,[double]$y,[double]$w,[double]$h,[double]$size=12,[string]$ink='#243443',[bool]$bold=$false,[bool]$center=$true) {
    $s = $slide.Shapes.AddTextbox(1,$x,$y,$w,$h)
    $s.Name = "MPKD_$name"
    $t = $s.TextFrame2
    $t.TextRange.Text = $text
    $t.TextRange.Font.Name = 'Arial'
    $t.TextRange.Font.Size = $size
    $t.TextRange.Font.Bold = $(if ($bold) {-1} else {0})
    $t.TextRange.Font.Fill.ForeColor.RGB = color $ink
    $t.TextRange.ParagraphFormat.Alignment = $(if ($center) {2} else {1})
    $t.VerticalAnchor = 3
    $t.MarginLeft = 0; $t.MarginRight = 0; $t.MarginTop = 0; $t.MarginBottom = 0
    $s
}
function segment([string]$name,[double]$x1,[double]$y1,[double]$x2,[double]$y2,[string]$ink='#34475A',[bool]$arrow=$false,[bool]$dashed=$false,[double]$weight=1.6) {
    $s = $slide.Shapes.AddLine($x1,$y1,$x2,$y2)
    $s.Name = "MPKD_$name"
    $s.Line.ForeColor.RGB = color $ink
    $s.Line.Weight = $weight
    if ($dashed) { $s.Line.DashStyle = 4 }
    if ($arrow) { $s.Line.EndArrowheadStyle = 3 }
    $s
}
function picture([string]$name,[string]$path,[double]$x,[double]$y,[double]$size) {
    $s = $slide.Shapes.AddPicture($path,0,-1,$x,$y,$size,$size)
    $s.Name = "MPKD_$name"
    $s
}
function chart([string]$name,[double]$x,[double]$y,[int[]]$bars,[string]$accent,[string]$fill) {
    box "${name}_frame" $x $y 70 63 $fill $accent 1.3 | Out-Null
    for ($i=0; $i -lt $bars.Count; $i++) {
        $bar = box "${name}_bar$i" ($x+9+$i*12) ($y+53-$bars[$i]) 8 $bars[$i] $accent $accent 0 | Out-Null
    }
}

$ink = '#263849'; $muted = '#637487'; $rule = '#D6E0E7'
$green = '#3B7B5E'; $blue = '#3F78A5'; $purple = '#75569B'; $red = '#C94C52'
$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$presentation = $null
try {
    $presentation = $ppt.Presentations.Open($target,0,0,0)
    if ($presentation.Slides.Count -ne 1) { throw 'Expected one slide; leaving existing content untouched.' }
    $existing = $presentation.Slides.Item(1).Shapes
    foreach ($shape in $existing) {
        if ($shape.Name -notlike 'MPKD_*') { throw 'Found a non-MPKD shape; leaving existing content untouched.' }
    }
    while ($existing.Count) { $existing.Item(1).Delete() }
    $presentation.PageSetup.SlideWidth = 1200
    $presentation.PageSetup.SlideHeight = 500
    $slide = $presentation.Slides.Item(1)

    # All connectors sit behind the editable boxes and text.
    segment 'input_split' 121 240 140 240 $ink | Out-Null
    segment 'view_bus' 140 156 140 333 $ink | Out-Null
    segment 'to_weak' 140 156 164 156 $ink $true | Out-Null
    segment 'to_strong' 140 333 164 333 $ink $true | Out-Null
    segment 'weak_to_models' 264 156 289 156 $ink | Out-Null
    segment 'strong_to_models' 264 333 289 333 $ink | Out-Null
    segment 'model_bus' 289 156 289 335 $ink | Out-Null
    segment 'to_teacher' 289 168 317 168 $ink $true | Out-Null
    segment 'to_student' 289 335 317 335 $ink $true | Out-Null
    segment 'teacher_output' 438 168 471 168 $ink $true | Out-Null
    segment 'student_output' 438 335 471 335 $ink $true | Out-Null

    segment 'offline_up' 438 124 462 124 $purple $false $true | Out-Null
    segment 'offline_rise' 462 124 462 83 $purple $false $true | Out-Null
    segment 'offline_bank' 462 83 699 83 $purple $true $true | Out-Null
    segment 'strong_feature' 438 215 699 215 $purple $true | Out-Null
    segment 'bank_to_distribution' 782 132 782 169 $purple $true | Out-Null
    segment 'distribution_to_gate' 865 209 899 209 $purple $true | Out-Null
    segment 'gate_to_reconstruction' 1030 209 1064 209 $purple $true | Out-Null

    segment 'original_target_down' 650 178 666 178 $muted | Out-Null
    segment 'original_target_rail' 666 178 666 262 $muted | Out-Null
    segment 'original_target_across' 666 262 1127 262 $muted | Out-Null
    segment 'original_target_to_reconstruction' 1127 262 1127 248 $muted $true | Out-Null
    segment 'reconstructed_down' 1127 248 1127 279 $red | Out-Null
    segment 'reconstructed_left' 1127 279 967 279 $red | Out-Null
    segment 'reconstructed_to_kd' 967 279 967 350 $red $true | Out-Null

    segment 'student_to_losses' 650 335 679 335 $blue | Out-Null
    segment 'student_loss_bus' 679 335 679 389 $blue | Out-Null
    segment 'student_to_original' 679 389 699 389 $blue $true | Out-Null
    segment 'student_to_strong' 679 335 881 335 $blue | Out-Null
    segment 'student_to_strong_down' 881 335 881 389 $blue | Out-Null
    segment 'student_strong_input' 881 389 894 389 $blue $true | Out-Null
    segment 'original_to_update' 855 390 1070 390 $ink $false | Out-Null
    segment 'strong_to_update' 1041 390 1070 390 $red $false | Out-Null
    segment 'update_input' 1070 390 1079 390 $ink $true | Out-Null

    label 'title' 'MPKD' 28 9 90 32 22 $ink $true $false | Out-Null
    label 'subtitle' 'Multi-center prototype knowledge distillation' 116 12 390 26 13 $muted $false $false | Out-Null
    segment 'top_rule' 28 49 1190 49 $rule $false $false 1 | Out-Null

    label 'input_label' 'Input' 28 181 94 18 13 $ink $true | Out-Null
    picture 'input_cat' $views[0] 28 199 94 | Out-Null
    label 'weak_label' 'Weak view' 165 84 99 19 13 $ink $true | Out-Null
    picture 'weak_cat' $views[1] 165 105 99 | Out-Null
    label 'strong_label' 'Strong view' 165 264 99 19 13 $ink $true | Out-Null
    picture 'strong_cat' $views[2] 165 285 99 | Out-Null

    box 'teacher_box' 318 123 120 92 '#ECF5EF' $green 2 | Out-Null
    label 'teacher' 'Teacher' 329 143 98 32 18 $green $true | Out-Null
    label 'teacher_frozen' 'frozen' 329 175 98 22 10 $green | Out-Null
    box 'student_box' 318 290 120 92 '#EAF3FB' $blue 2 | Out-Null
    label 'student' 'Student' 329 311 98 32 18 $blue $true | Out-Null
    label 'student_trainable' 'trainable' 329 343 98 22 10 $blue | Out-Null

    chart 'teacher_weak' 486 126 @(18,30,43,24,16) '#6B9DC8' '#EDF5FB'
    chart 'teacher_strong' 580 126 @(15,27,36,29,19) '#C97779' '#FFF2F2'
    label 'teacher_weak_label' 'weak target' 482 105 78 18 10 $muted | Out-Null
    label 'teacher_strong_label' 'strong target' 575 105 80 18 10 $muted | Out-Null
    label 'teacher_weak_symbol' 'p_w^T' 486 192 70 18 11 $ink $true | Out-Null
    label 'teacher_strong_symbol' 'p_s^T' 580 192 70 18 11 $ink $true | Out-Null
    chart 'student_weak' 486 294 @(17,27,40,26,18) '#6B9DC8' '#EDF5FB'
    chart 'student_strong' 580 294 @(14,25,34,39,26) '#C97779' '#FFF2F2'
    label 'student_weak_symbol' 'z_w^S' 486 359 70 18 11 $ink $true | Out-Null
    label 'student_strong_symbol' 'z_s^S' 580 359 70 18 11 $ink $true | Out-Null

    box 'prototype_bank' 700 68 165 65 '#F6F1FA' $purple 1.7 $true | Out-Null
    label 'prototype_bank_title' 'Fixed prototypes' 708 76 149 23 14 $purple $true | Out-Null
    label 'prototype_bank_subtitle' 'Reliable features / K-means' 708 99 149 18 9 $muted | Out-Null
    foreach ($i in 0..2) {
        $d = $slide.Shapes.AddShape(9,(747+$i*17),119,8,8)
        $d.Name = "MPKD_prototype_dot$i"
        $d.Fill.ForeColor.RGB = color $purple
        $d.Line.Visible = 0
    }
    box 'prototype_distribution' 700 170 165 78 '#F6F1FA' $purple 1.7 | Out-Null
    label 'prototype_distribution_title' 'Prototype distribution' 707 183 151 24 13 $purple $true | Out-Null
    label 'aggregation' 'Log-Mean-Exp' 707 213 151 18 10 $muted | Out-Null
    box 'gate_box' 900 170 130 78 '#F6F1FA' $purple 1.7 | Out-Null
    label 'gate_title' 'Reliability gate' 905 183 120 24 13 $purple $true | Out-Null
    $gateSymbols = 'u   a   r   q   ' + [char]0x2192 + '   ' + [char]0x03BB
    label 'gate_inputs' $gateSymbols 905 213 120 18 10 $purple $true | Out-Null
    box 'reconstruction_box' 1065 170 125 78 '#FFF0F0' $red 1.9 | Out-Null
    label 'reconstruction_title' 'Strong target' 1072 180 111 27 13 $red $true | Out-Null
    label 'reconstruction_subtitle' 'reconstruction' 1072 207 111 22 12 $red $true | Out-Null

    label 'distillation_label' 'DISTILLATION' 700 312 150 18 10 $muted $true $false | Out-Null
    segment 'distillation_rule' 700 332 1190 332 $rule $false $false 1 | Out-Null
    box 'original_kd_box' 700 351 155 77 '#F4F7F9' $rule 1.3 | Out-Null
    label 'original_kd_title' 'Original KD' 709 363 137 25 13 $ink $true | Out-Null
    label 'original_kd_subtitle' 'weak-weak / cross-view' 709 392 137 18 10 $muted | Out-Null
    box 'strong_kd_box' 894 351 147 77 '#FFF0F0' $red 1.7 | Out-Null
    label 'strong_kd_title' 'Strong-strong KD' 901 363 133 25 13 $red $true | Out-Null
    label 'strong_kd_subtitle' 'reconstructed target' 901 392 133 18 10 $red | Out-Null
    box 'update_box' 1079 351 111 77 '#EAF3FB' $blue 1.7 | Out-Null
    label 'update_title' 'Update student' 1084 363 101 25 12 $blue $true | Out-Null
    label 'update_subtitle' 'CE + KD' 1084 392 101 18 10 $blue | Out-Null

    $presentation.Save()
    $slide.Export((Join-Path $PSScriptRoot 'mpkd_ppt_preview.png'),'PNG',2400,1000) | Out-Null
    Write-Output "Saved $target ($($slide.Shapes.Count) editable shapes)"
} finally {
    if ($presentation) { $presentation.Close() }
    $ppt.Quit()
    foreach ($path in $views) { if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path } }
}
