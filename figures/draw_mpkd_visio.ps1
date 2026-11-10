$ErrorActionPreference = 'Stop'

try { $visio = [Runtime.InteropServices.Marshal]::GetActiveObject('Visio.Application') }
catch { $visio = New-Object -ComObject Visio.Application; $visio.Visible = $true }
$out = Join-Path $PSScriptRoot 'mpkd_method_overview_clean.vsdx'
$preview = Join-Path $PSScriptRoot 'mpkd_method_overview_clean_preview.png'
$doc = @($visio.Documents | Where-Object { $_.FullName -eq $out } | Select-Object -First 1)[0]
if (-not $doc) { $doc = $visio.Documents.Add('') }
$page = $doc.Pages.Item(1)
$unowned = @($page.Shapes | Where-Object { $_.NameU -notlike 'MPKD_*' })
if ($unowned.Count) { throw 'The Visio page has user shapes; refusing to overwrite them.' }
while ($page.Shapes.Count) { $page.Shapes.Item(1).Delete() | Out-Null }
$page.PageSheet.CellsU('PageWidth').FormulaU = '16.8 in'
$page.PageSheet.CellsU('PageHeight').FormulaU = '5.6 in'
$fontId = ($doc.Fonts | Where-Object { $_.Name -eq 'Arial' } | Select-Object -First 1).ID
$script:shapeId = 0

function rgb([string]$hex) {
    $hex = $hex.TrimStart('#')
    'RGB({0},{1},{2})' -f [Convert]::ToInt32($hex.Substring(0,2),16),
        [Convert]::ToInt32($hex.Substring(2,2),16),
        [Convert]::ToInt32($hex.Substring(4,2),16)
}
function setCell($shape, [string]$name, [string]$formula) {
    $shape.CellsU($name).FormulaU = $formula
}
function rect($x1,$y1,$x2,$y2,[string]$fill,[string]$stroke,[double]$width=0.018,[bool]$dash=$false) {
    $s=$page.DrawRectangle($x1,$y1,$x2,$y2)
    $script:shapeId++
    $s.NameU="MPKD_$script:shapeId"
    if ($fill) { setCell $s 'FillForegnd' (rgb $fill); setCell $s 'FillPattern' '1' }
    else { setCell $s 'FillPattern' '0' }
    if ($stroke) {
        setCell $s 'LineColor' (rgb $stroke)
        setCell $s 'LineWeight' "$width in"
        setCell $s 'LinePattern' $(if ($dash) { '2' } else { '1' })
    } else { setCell $s 'LinePattern' '0' }
    $s
}
function txt($x1,$y1,$x2,$y2,[string]$value,[double]$size=11,[string]$ink='#1B2533',[bool]$bold=$false,[int]$align=1) {
    $s=rect $x1 $y1 $x2 $y2 '' ''
    $s.Text=$value
    setCell $s 'Char.Font' "$fontId"
    setCell $s 'Char.Size' "$size pt"
    setCell $s 'Char.Color' (rgb $ink)
    setCell $s 'Char.Style' $(if ($bold) { '1' } else { '0' })
    setCell $s 'Para.HorzAlign' "$align"
    setCell $s 'VerticalAlign' '1'
    $s
}
function line($x1,$y1,$x2,$y2,[string]$ink='#293647',[bool]$arrow=$true,[bool]$dash=$false,[double]$width=0.022) {
    $s=$page.DrawLine($x1,$y1,$x2,$y2)
    $script:shapeId++
    $s.NameU="MPKD_$script:shapeId"
    setCell $s 'LineColor' (rgb $ink)
    setCell $s 'LineWeight' "$width in"
    setCell $s 'LinePattern' $(if ($dash) { '2' } else { '1' })
    if ($arrow) { setCell $s 'EndArrow' '4' }
    $s
}
function dot($x,$y,[string]$fill,[double]$diameter=0.15) {
    $s=$page.DrawOval($x,$y,($x+$diameter),($y+$diameter))
    $script:shapeId++
    $s.NameU="MPKD_$script:shapeId"
    setCell $s 'FillForegnd' (rgb $fill)
    setCell $s 'LinePattern' '0'
    $s
}
function histogram($x,$y,[double[]]$heights,[string]$fill,[string]$stroke) {
    rect $x $y ($x+1.04) ($y+0.73) $fill $stroke 0.016 | Out-Null
    $colors=@('#6D9FD1','#9CC0E5','#4E7FAE','#9CC0E5','#6D9FD1')
    for ($i=0;$i -lt $heights.Count;$i++) {
        rect ($x+0.14+$i*0.175) ($y+0.13) ($x+0.27+$i*0.175) ($y+0.13+$heights[$i]) $colors[$i] '' | Out-Null
    }
}
function photo($path,$x1,$y1,$x2,$y2) {
    $s=$page.Import($path)
    $script:shapeId++
    $s.NameU="MPKD_$script:shapeId"
    setCell $s 'Width' "$(($x2-$x1)) in"
    setCell $s 'Height' "$(($y2-$y1)) in"
    setCell $s 'PinX' "$(($x1+$x2)/2) in"
    setCell $s 'PinY' "$(($y1+$y2)/2) in"
    $s
}

# The supplied example contains three distinct augmented views of one sample.
# Crop only those photographs; all diagram text, geometry, and arrows are Visio shapes.
Add-Type -AssemblyName System.Drawing
$reference=Join-Path (Split-Path $PSScriptRoot -Parent) 'fig/framework.png'
$bitmap=[Drawing.Bitmap]::FromFile($reference)
$cropped=@()
$regions=@(
    @(17,399,315,313),
    @(439,200,313,313),
    @(439,642,315,313)
)
try {
    foreach ($r in $regions) {
        $crop=New-Object Drawing.Bitmap $r[2],$r[3]
        $graphics=[Drawing.Graphics]::FromImage($crop)
        $source=New-Object Drawing.Rectangle $r[0],$r[1],$r[2],$r[3]
        $dest=New-Object Drawing.Rectangle 0,0,$r[2],$r[3]
        $graphics.DrawImage($bitmap,$dest,$source,[Drawing.GraphicsUnit]::Pixel)
        $graphics.Dispose()
        $path=Join-Path $env:TEMP ("mpkd_visio_view_$([guid]::NewGuid().ToString('N')).png")
        $crop.Save($path,[Drawing.Imaging.ImageFormat]::Png)
        $crop.Dispose()
        $cropped+=,$path
    }
} finally { $bitmap.Dispose() }

$ink='#233143'; $soft='#718091'; $rule='#D4DDE5'
$green='#35775A'; $greenFill='#EAF4ED'
$blue='#326EA4'; $blueFill='#EAF3FB'
$violet='#7653A4'; $violetFill='#F4EFF9'
$coral='#C34E52'; $coralFill='#FFF0F0'

# The white page background keeps the exported preview at the intended page size.
$background=rect 0 0 16.8 5.6 '#FFFFFF' ''

# Sparse orthogonal connectors, drawn behind the modules.
line 1.73 3.31 2.00 3.31 $ink $false | Out-Null
line 2.00 2.35 2.00 5.17 $ink $false | Out-Null
line 2.00 5.17 2.22 5.17 $ink | Out-Null
line 2.00 2.35 2.22 2.35 $ink | Out-Null
line 3.72 5.17 3.98 5.17 $ink $false | Out-Null
line 3.72 2.35 3.98 2.35 $ink $false | Out-Null
line 3.98 2.35 3.98 5.17 $ink $false | Out-Null
line 3.98 4.90 4.44 4.90 $ink | Out-Null
line 3.98 2.60 4.44 2.60 $ink | Out-Null
line 6.36 4.89 7.02 4.89 $ink | Out-Null
line 6.36 2.60 7.02 2.60 $ink | Out-Null

# Weak teacher features build a fixed prototype bank; strong features query it.
line 6.36 5.36 6.63 5.36 $violet $false $true 0.018 | Out-Null
line 6.63 5.36 6.63 6.22 $violet $false $true 0.018 | Out-Null
line 6.63 6.22 9.67 6.22 $violet $true $true 0.018 | Out-Null
line 6.36 4.40 6.69 4.40 $violet $false | Out-Null
line 6.69 4.40 6.69 3.69 $violet $false | Out-Null
line 6.69 3.69 10.72 3.69 $violet $false | Out-Null
line 10.72 3.69 10.72 3.95 $violet | Out-Null
line 10.76 5.52 10.76 4.95 $violet | Out-Null
line 11.91 4.45 12.30 4.45 $violet | Out-Null
line 14.14 4.45 14.49 4.45 $violet | Out-Null

# Original strong teacher target is a single gray rail over the MPKD blocks.
line 9.22 5.03 9.40 5.03 $soft $false $false 0.016 | Out-Null
line 9.40 5.03 9.40 5.20 $soft $false $false 0.016 | Out-Null
line 9.40 5.20 15.56 5.20 $soft $false $false 0.016 | Out-Null
line 15.56 5.20 15.56 4.95 $soft $true $false 0.016 | Out-Null

# Reconstructed target changes only the same-view strong KD path.
line 15.55 3.95 15.55 3.38 $coral $false $false 0.030 | Out-Null
line 15.55 3.38 13.32 3.38 $coral $false $false 0.030 | Out-Null
line 13.32 3.38 13.32 2.92 $coral $true $false 0.030 | Out-Null
line 9.22 2.60 9.67 2.60 $blue | Out-Null
line 11.91 2.50 12.10 2.50 $ink $false | Out-Null
line 12.10 2.50 12.10 1.84 $ink $false | Out-Null
line 12.10 1.84 15.57 1.84 $ink $false | Out-Null
line 15.57 1.84 15.57 2.08 $ink | Out-Null
line 14.34 2.50 14.68 2.50 $ink | Out-Null

# Source and perturbation views use example photographs, but with clean framing.
txt 0.38 4.12 1.72 4.48 'Input' 14 $ink $true | Out-Null
photo $cropped[0] 0.38 2.64 1.73 4.00 | Out-Null
txt 2.22 5.92 3.72 6.25 'Weak view' 14 $ink $true | Out-Null
photo $cropped[1] 2.22 4.43 3.72 5.89 | Out-Null
txt 2.22 3.09 3.72 3.42 'Strong view' 14 $ink $true | Out-Null
photo $cropped[2] 2.22 1.61 3.72 3.07 | Out-Null

# Two model lanes, each aligned with its weak and strong predictions.
rect 4.44 4.30 6.36 5.55 $greenFill $green 0.025 | Out-Null
txt 4.64 4.73 6.15 5.19 'Teacher' 21 $green $true | Out-Null
txt 4.72 4.42 6.08 4.68 'frozen' 10 $green | Out-Null
rect 4.44 2.00 6.36 3.25 $blueFill $blue 0.025 | Out-Null
txt 4.64 2.43 6.15 2.89 'Student' 21 $blue $true | Out-Null
txt 4.72 2.11 6.08 2.37 'trainable' 10 $blue | Out-Null

txt 7.02 5.60 8.06 5.87 'weak target' 10 $soft | Out-Null
txt 8.18 5.60 9.22 5.87 'strong target' 10 $soft | Out-Null
histogram 7.02 4.54 @(0.20,0.36,0.48,0.22,0.17) '#E9F3FB' '#6994BB'
histogram 8.18 4.54 @(0.18,0.31,0.39,0.26,0.22) '#FFF0F0' '#D3878A'
txt 7.03 4.22 8.05 4.48 'p_w^T' 10 $ink $true | Out-Null
txt 8.19 4.22 9.21 4.48 'p_s^T' 10 $ink $true | Out-Null

txt 7.02 3.31 8.06 3.58 'weak output' 10 $soft | Out-Null
txt 8.18 3.31 9.22 3.58 'strong output' 10 $soft | Out-Null
histogram 7.02 2.25 @(0.18,0.30,0.42,0.25,0.21) '#E9F3FB' '#6994BB'
histogram 8.18 2.25 @(0.14,0.26,0.34,0.36,0.28) '#FFF0F0' '#D3878A'
txt 7.03 1.94 8.05 2.20 'p_w^S' 10 $ink $true | Out-Null
txt 8.19 1.94 9.21 2.20 'p_s^S' 10 $ink $true | Out-Null

# The small dashed bank is the only offline branch.
rect 9.67 5.52 11.91 6.58 $violetFill $violet 0.019 $true | Out-Null
txt 9.82 6.19 11.76 6.48 'Fixed prototypes' 13 $violet $true | Out-Null
txt 9.82 5.91 11.76 6.16 'Reliable features  /  K-means' 9 $soft | Out-Null
dot 10.17 5.65 '#70549D' | Out-Null
dot 10.63 5.65 '#9979BA' | Out-Null
dot 11.09 5.65 '#B9A2CD' | Out-Null

rect 9.67 3.95 11.91 4.95 $violetFill $violet 0.019 | Out-Null
txt 9.79 4.54 11.78 4.87 'Prototype distribution' 13 $violet $true | Out-Null
txt 9.82 4.12 11.76 4.43 'Log-Mean-Exp' 10 $soft | Out-Null

rect 12.30 3.95 14.14 4.95 $violetFill $violet 0.019 | Out-Null
txt 12.41 4.57 14.03 4.86 'Reliability gate' 13 $violet $true | Out-Null
$gateSymbols='u   a   r   q    ' + [char]0x2192 + '    ' + [char]0x03BB
txt 12.41 4.22 14.03 4.50 $gateSymbols 10 $violet $true | Out-Null

rect 14.49 3.95 16.48 4.95 $coralFill $coral 0.024 | Out-Null
txt 14.60 4.55 16.36 4.86 'Strong target' 13 $coral $true | Out-Null
txt 14.60 4.15 16.36 4.45 'reconstruction' 12 $coral $true | Out-Null

# A light objective strip keeps existing CRLD supervision distinct from MPKD.
line 9.67 3.14 16.48 3.14 $rule $false $false 0.011 | Out-Null
txt 9.67 2.98 16.48 3.16 'DISTILLATION' 9 $soft $true 0 | Out-Null
rect 9.67 2.08 11.91 2.92 '#F4F7F9' $rule 0.015 | Out-Null
txt 9.78 2.47 11.80 2.76 'Original CRLD' 12 $ink $true | Out-Null
txt 9.78 2.20 11.80 2.43 'weak-weak / cross-view' 9 $soft | Out-Null
rect 12.30 2.08 14.34 2.92 $coralFill $coral 0.020 | Out-Null
txt 12.40 2.45 14.24 2.75 'Strong-strong KD' 12 $coral $true | Out-Null
txt 12.40 2.19 14.24 2.43 'corrected target' 9 $coral | Out-Null
rect 14.68 2.08 16.48 2.92 $blueFill $blue 0.018 | Out-Null
txt 14.79 2.47 16.37 2.75 'Update student' 12 $blue $true | Out-Null
txt 14.79 2.19 16.37 2.43 'CE + KD' 9 $blue | Out-Null

try {
    foreach ($shape in $page.Shapes) {
        if ($shape.ID -ne $background.ID) {
            $y=$shape.CellsU('PinY').ResultIU - 1.25
            setCell $shape 'PinY' "$y in"
        }
    }
    if ($doc.FullName -eq $out) { $doc.Save() | Out-Null }
    else { $doc.SaveAs($out) | Out-Null }
    $page.Export($preview) | Out-Null
    Write-Host "Saved $out ($($page.Shapes.Count) native Visio shapes)"
    Write-Host "Preview $preview"
} finally {
    foreach ($path in $cropped) { Remove-Item -LiteralPath $path -ErrorAction SilentlyContinue }
}
