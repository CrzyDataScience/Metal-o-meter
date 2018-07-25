

# Set some paths
$SoXLocation = "C:\Program Files (x86)\sox-14-4-2\sox.exe"

$MainPath = "D:\Projects\CrazyDataScience\Metal-o-meter"
$RawMP3Path = "D:\Projects\CrazyDataScience\Metal-o-meter\Raw"
$SpectroPath = "D:\Projects\CrazyDataScience\Metal-o-meter\Spectrograms"
$SlicePath = "D:\Projects\CrazyDataScience\Metal-o-meter\Spectrograms\Slices"
$TempPath = "D:\Projects\CrazyDataScience\Metal-o-meter\Temp"


# Variables
$CreateMonoFiles = 1

$shell = New-Object -ComObject "Shell.Application"

# Get all the mp3 files from the Raw folder
$ObjDir = $shell.NameSpace($RawMP3Path)
$Files = Get-ChildItem $RawMP3Path | Where-Object {$_.Extension -eq ".mp3"}

$i = 1

Foreach($File in $Files)
    {
    $ObjFile = $ObjDir.parsename($File.Name)
    $MetaData = @{}
    $MP3 = ($ObjDir.Items()|?{$_.path -like "*.mp3"})
    $PropertArray = 0,1,2,12,13,14,15,16,17,18,19,20,21,22,27,28,36,220,223        

    Foreach($item in $PropertArray)
        { 
        If($ObjDir.GetDetailsOf($ObjFile, $item)) #To avoid empty values
            {
            $MetaData[$($ObjDir.GetDetailsOf($MP3,$item))] = $ObjDir.GetDetailsOf($ObjFile, $item)
            }
        }
     
    If ($CreateMonoFiles -eq 1)
        {          
        # Create a mono file from the MP3 (can take some time depending on the number of files)
        $SoxCreateMonoCommand = "$RawMP3Path\$File $TempPath\$File remix 1,2"
        $SoxCreateMonoCommand = $SoxCreateMonoCommand.Split(" ")
        & "$SoXLocation" $SoxCreateMonoCommand
        }

    $ShortFileName = $File.BaseName    

    # Create the spectrograms from the mono files
    $SoxCreateSpectrogram = "$TempPath\$File -n spectrogram -Y 200 -X 50 -m -r -o $SpectroPath\$ShortFileName.png"
    $SoxCreateSpectrogram = $SoxCreateSpectrogram.Split(" ")
    & "$SoxLocation" $SoxCreateSpectrogram

    $i = $i + 1

    }

# Run Python script to slice the spectrograms
& python slice_spectograms.py D:/Projects/CrazyDataScience/Metal-o-meter/Spectrograms D:/Projects/CrazyDataScience/Metal-o-meter/Spectrograms/Slices
