from PIL import Image
import os.path
import csv
import sys

if __name__ == "__main__":
    spec_path = str(sys.argv[1])
    spec_slice_path = str(sys.argv[2])

spec_slice_path = spec_slice_path + "/"
	
# For each file in the Spectrograms folder
for filename in os.listdir(spec_path):
	if filename.endswith(".png"):
        
            filebase = os.path.splitext(filename)[0]
			
            img = Image.open("D:/Projects/CrazyDataScience/Metal-o-meter/Spectrograms/"+filename)
            
            width, height = img.size
            nbSamples = int(width/128)
            width - 128

			# If the slices folder does not exist, create it
            if not os.path.exists(os.path.dirname(spec_slice_path)):
                try:
                    os.makedirs(os.path.dirname(spec_slice_path))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

            #For each sample
            for i in range(nbSamples):
                print ("Processing slice nr: ", (i+1), "/", nbSamples, "for file ", filename)
                
                startPixel = i*128
                imgTmp = img.crop((startPixel, 1, startPixel + 128, 128 + 1))
                imgTmp.save(spec_slice_path + filebase + "_slice_" + str(i) + ".png")
