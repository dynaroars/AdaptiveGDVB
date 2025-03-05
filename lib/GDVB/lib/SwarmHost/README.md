# SwarmHost
Spawns Locusts.



## 1 INSTALLATION

### I. Install environment
Install the conda environment
`conda env create --name SwarmHost -f env.yml`

### II. Install verifiers
Clone your preferred verifier(`VERIFIER`) into the `lib` directory.
`mkdir lib`
`cd lib`
`git clone [VERIFIER_REPO_LINK]`
e.g.,
`git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN`

Install the conda environment of your preferred verifier(`VERIFIER`)
`conda env create --name [VERIFIER] -f envs/[VERIFIER].yml`
e.g.,
`conda env create --name abcrown -f envs/abcrown.yml` 
    


#### a) Supported verifiers
A list of supported [VERIFIER]s and their [VERIFIER_REPO_LINK]s are shown as follows:
    
> [$\alpha$-$\beta$-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
> [mnbab]
> [nnenum]
> [verinet]
> [neuralsat]
> [veristable]

### III. Get miscellaneous tools
Download the resource monitor from the DNNV framework.
`wget https://github.com/dlshriver/dnnv/blob/main/tools/resmonitor.py lib\`


### 2 USAGE

#### I. Prepare your neural network model and property

##### a) neural network model format
This tool supports the standard ONNX model format defined in [VNN-LIB](https://www.vnnlib.org). Prepare your neural network model in this format.

##### b) property
This tool also uses the VNNLIB standard(.vnnlib) for the property language. You can either prepare your own properties by following the standard or use the built-in LocalRobustness properties generator.

##### c) a list of other options
> [--time]
> [--memory]