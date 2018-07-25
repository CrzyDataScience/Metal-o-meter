# Metal-o-meter
This repository contains all the example code for the Crazy Data Science - Metal-o-meter project.
(https://youtu.be/OoyTT4nEm_E)


## Reproducing the examples from the start

- Download SoX for your operating system
- Create the following folder structure:
  - Model
  - Raw
    - tmp
  - Spectrograms
    - Slices
	    - Test
		- Train
		   - 01_Non_metal
		   - 02_Metal
  - tflogs
  
- Place MP3 files in the Raw folder, make sure they are labeled correctly (Non-metal or Metal)
- Run the GetMP3MetaData.ps1 script, make sure paths are configured correctly before executing the code
- If all is well, spectrogram slices should be created in the Spectrogram\Slices directory
- Move the slices into the correct genre subfolder in the Train folder when you want to train the model, if you want to predict the genre of the spectrograms you can place them in the Test folder.
- Run the code in the Metal-o-meter.py script or better yet, import it into a Jupyter notebook so you can go through the code one step at a time.
  Again, make sure al paths are configured to the correct folders on your system.
  
  