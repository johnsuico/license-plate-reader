# license-plate-reader
CMPE 188 Project. The goal is to create an application that will take in user images of license plates and the application will be able to read the license plate.

## Python Requirements
- This was tested using [Python 3.8](https://www.python.org/downloads/release/python-380/)
Feel free to try and use the latest version of Python (3.9). At the time of writing this, only tested with Python 3.8

- Might need to uninstall all other versions of python to be able to follow the instructions.

- [Download Git](https://git-scm.com/downloads)
  - If you don't have it already
  - To clone and push to the repo

## Cloning the directory
- Open CMD/Powershell/Terminal
  - CD into the directory of your choice (we'll be using desktop)\

  ``` cd desktop ```
  - Clone this repo

  ``` git clone https://github.com/johnsuico/license-plate-reader.git ```

## Installing pip packages
- Install numpy

  ``` pip install numpy ```

- Install OpenCV

  ``` pip install opencv-python ```

## Running the program
At the moment of writing this, there is only one python file. "cannystill.py"
This file just takes an image and gets the canny edges and returns the image to the window

- Go into the license-plate-reader directory

  ``` cd license-plate-reader ```

- Run "cannystill.py"

  ``` py cannystill.py ```