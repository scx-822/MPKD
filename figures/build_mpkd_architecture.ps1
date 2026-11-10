param(
    [string]$SourceImage = (Join-Path $PSScriptRoot '..\data\tiny-imagenet-200\val\n07734744\images\val_9778.JPEG'),
    [string]$Output = (Join-Path $PSScriptRoot 'mpkd_architecture.svg'),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
if ((Test-Path -LiteralPath $Output) -and -not $Force) {
    throw "Output exists: $Output. Pass -Force only when replacing a generated version."
}

Add-Type -AssemblyName System.Drawing
$source = [Drawing.Bitmap]::FromFile((Resolve-Path -LiteralPath $SourceImage).Path)
$svg = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'mpkd_architecture.template.svg'), [Text.Encoding]::UTF8)
$views = @(
    @{ token = '__INPUT_IMAGE__'; box = [Drawing.Rectangle]::new(0, 0, 64, 64); flip = $false; strong = $false },
    @{ token = '__WEAK_IMAGE__'; box = [Drawing.Rectangle]::new(3, 3, 58, 58); flip = $true; strong = $false },
    @{ token = '__STRONG_IMAGE__'; box = [Drawing.Rectangle]::new(7, 7, 50, 50); flip = $false; strong = $true }
)

try {
    if ($source.Width -ne 64 -or $source.Height -ne 64) {
        throw "Expected a 64x64 Tiny-ImageNet sample, got $($source.Width)x$($source.Height)."
    }
    foreach ($view in $views) {
        $box = $view.box
        $crop = [Drawing.Bitmap]::new(256, 256)
        try {
            $graphics = [Drawing.Graphics]::FromImage($crop)
            try {
                $graphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $dest = [Drawing.Rectangle]::new(0, 0, 256, 256)
                if ($view.strong) {
                    $attributes = [Drawing.Imaging.ImageAttributes]::new()
                    try {
                        $color = [Drawing.Imaging.ColorMatrix]::new()
                        $color.Matrix00 = 1.18
                        $color.Matrix11 = 1.08
                        $color.Matrix22 = 0.76
                        $color.Matrix40 = -0.035
                        $attributes.SetColorMatrix($color)
                        $graphics.DrawImage($source, $dest, $box.X, $box.Y, $box.Width, $box.Height, [Drawing.GraphicsUnit]::Pixel, $attributes)
                    } finally { $attributes.Dispose() }
                    $brush = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(136, 142, 138))
                    try { $graphics.FillRectangle($brush, 194, 180, 38, 42) }
                    finally { $brush.Dispose() }
                } else {
                    $graphics.DrawImage($source, $dest, $box, [Drawing.GraphicsUnit]::Pixel)
                }
            } finally { $graphics.Dispose() }
            if ($view.flip) { $crop.RotateFlip([Drawing.RotateFlipType]::RotateNoneFlipX) }
            $memory = [IO.MemoryStream]::new()
            try {
                $crop.Save($memory, [Drawing.Imaging.ImageFormat]::Png)
                $uri = 'data:image/png;base64,' + [Convert]::ToBase64String($memory.ToArray())
                if (-not $svg.Contains($view.token)) { throw "Missing template token: $($view.token)" }
                $svg = $svg.Replace($view.token, $uri)
            } finally { $memory.Dispose() }
        } finally { $crop.Dispose() }
    }
} finally { $source.Dispose() }

$null = [xml]$svg
[IO.File]::WriteAllText($Output, $svg, [Text.UTF8Encoding]::new($false))
Write-Output "Generated $Output"
