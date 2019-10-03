# Lanzador de OneGeo
import time
import os
import sys
import pdb


def main():
    # Importando archivos para que se ejecute el programa
    # Importando main_server
    from dependencias.main_server import main

    # Corriendo la funcion principal de main_server
    main()

    time.sleep(0.9)
    # pdb.set_trace()
    os.system("./dependencias/start.sh")


if __name__ == "__main__":
    main()
