</center> <center> <h1>PhD Repository</h1> </center>

Overview of PhD code

## PhD Title:  Diabatic Eddies in idealised channel models ##

This work used the MITgcm to simulate over 40 different experiments of idealised channel models altering the northern boundary condition. This repository gives an overview of the code used for this work.

1. **[MITgcm template setup](https://github.com/hlburns/PhD/wiki/Model-Setup)**
    * This is a template of channel model set up, input files, header files, optimization, job submission.

2. **NetCDF manipulation**
    * A large amount of data is generated and this must be layed out in a senisble matter

3. **Python modules for analysis of MITgcm output**
    * Custom base modules.

4. **Ipython Notebooks**
    * For plotting and bespoke caculations.
    
5. **Selected Figures**
      * Mainly used for illustration in the wiki, but gives an overview of results and productions of some python scripts

## Usage ##

This repository can be cloned to provide all the code and tools required to reproduce my PhD work and continue on from it. The MITgcm setup directory contains the code for running the channel model and the manipulation and python folders contain the code to analyse and visulise some of the channel model physics outlined in the wiki.

## Documentation ##

This project is documented thoroughly through the repository [wiki](https://github.com/hlburns/Channel_model_complete/wiki). There you will find detailed information to both the original science and code.

## Requirements ##

1. [MITgcm model code](https://github.com/MITgcm/MITgcm)

2. [Python 2.7](https://www.anaconda.com/download/)

3. [NCO](http://nco.sourceforge.net/)

4. A HPC system to run the model!

## License ##

This project is licenced under the [MIT license](https://choosealicense.com/licenses/mit/)
