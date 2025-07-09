import logging
import os
from datetime import datetime

def setup_logger(name, log_file="app.log", level=logging.INFO):
    """
    Configura y retorna un logger para un nombre específico,
    escribiendo logs en un archivo y en la consola.

    Args:
        name (str): El nombre del logger (típicamente __name__ del módulo que lo usa).
        log_file (str): El nombre del archivo donde se guardarán los logs.
        level (int): El nivel mínimo de los logs a registrar (ej. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: El objeto logger configurado.
    """
    # Crear un directorio 'logs' si no existe
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Configurar el handler para escribir en archivo
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # Configurar el handler para imprimir en consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar añadir handlers múltiples veces si el logger ya existe
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Ejemplo de uso si se ejecuta este archivo directamente para prueba
if __name__ == "__main__":
    test_logger = setup_logger("test_module")
    test_logger.debug("Este es un mensaje de depuración.")
    test_logger.info("Este es un mensaje informativo.")
    test_logger.warning("Esta es una advertencia.")
    test_logger.error("Este es un mensaje de error.")
    test_logger.critical("Este es un mensaje crítico.")

    # Puedes probar con otro logger para ver cómo se diferencian
    another_logger = setup_logger("another_module", log_file="another_app.log", level=logging.DEBUG)
    another_logger.info("Este es un mensaje informativo de otro módulo.")
    another_logger.debug("Este es un mensaje de depuración de otro módulo.")