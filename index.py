# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:20:40 2018

@author: DELL
"""

import MainProgram as start


def main():  
    ## Assign source with reviews of a hotel received from user input
    choice=1
    if choice==0:
     result=start.find(source,0)
    else:
     result=start.find(source,1) 
     
    print(result)
 
if __name__ == "__main__": main()   
    