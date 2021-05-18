# pytorch_tut_00



[Difference between Conda & Pip?](https://www.anaconda.com/blog/understanding-conda-and-pip#:~:text=Pip%20installs%20Python%20packages%20whereas,software%20written%20in%20any%20language.&text=Another%20key%20difference%20between%20the,the%20packages%20installed%20in%20them.)  

TLDR;  
Conda more like Homebrew but it's cross platform
[more here](https://towardsdatascience.com/managing-project-specific-environments-with-conda-b8b50aa8be0e) 
Inc Conda vs MiniConda vs Anaconda.  

Pip is the Python Packaging Authority’s recommended tool for installing packages from the Python Package Index, PyPI. Pip installs Python software packaged as wheels or source distributions. The latter may require that the system have compatible compilers, and possibly libraries, installed before invoking pip to succeed.

Conda is a cross platform package and environment manager that installs and manages conda packages from the Anaconda repository as well as from the Anaconda Cloud. Conda packages are binaries. There is never a need to have compilers available to install them. Additionally conda packages are not limited to Python software. They may also contain C or C++ libraries, R packages or any other software.


### Installing conda on osx:
DLOAD miniconda installer [from here](https://conda.io/miniconda.html) - used python 3.9 bash.  
> shasum -a 256 /Users/simon/Downloads/Miniconda3-py39_4.9.2-MacOSX-x86_64.sh  
b3bf77cbb81ee235ec6858146a2a84d20f8ecdeb614678030c39baacb5acbed1  /Users/simon/Downloads/Miniconda3-py39_4.9.2-MacOSX-x86_64.sh  
SHA256 hash from dload link.  
b3bf77cbb81ee235ec6858146a2a84d20f8ecdeb614678030c39baacb5acbed1.  
Match!  

Install:
> bash /Users/simon/Downloads/Miniconda3-py39_4.9.2-MacOSX-x86_64.sh
Accept licence
Miniconda3 will now be installed into this location:
/Users/simon/miniconda3
Say yes to initialise, start new shell
> conda --version
conda 4.9.2

### Next Pytorch...   
[Pytorch Home Page](https://pytorch.org/).  
![install config](https://github.com/UnacceptableBehaviour/pytorch_tut_00/blob/main/imgs/pythorch_install_config.png).  









