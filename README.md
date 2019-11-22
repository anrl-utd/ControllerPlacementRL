# "Repo for Leader Election Problem for ANRL" 

## Installation
To install everything needed to run this, run the below command:

    pip install -e .
This installs openai-gym and inserts the custom environment.

## List of environments
 - **Controller-v0**
 -- Intended to be the basic implementations without graph kernels and other optimizations
 - **Controller-RandomStart-v0**
 -- Environment that "nudges" controllers (starting with random placement of controllers in each cluster) to adjacent nodes