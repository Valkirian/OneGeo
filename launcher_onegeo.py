# Lanzador de OneGeo
import time
import os
import sys
import pdb


def main():
    # Importando archivos para que se ejecute el programa
    # Importando main_server
    from components.thinsec.microscope import main_server

    # Corriendo la funcion principal de main_server
    main()
    time.sleep(0.9)
    # pdb.set_trace()
    os.system("./components/start.sh")


if __name__ == "__main__":
    main()
