import os
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import logging



# --- Configuración del Logging ---
from log_manager import setup_logger

logger = setup_logger(__name__)


def generate_ssh_keys(private_key_path="id_rsa_erp", public_key_path="id_rsa_erp.pub", passphrase=None):
    """
    Genera un par de claves SSH (RSA) y las guarda en archivos.

    Args:
        private_key_path (str): Ruta para guardar la clave privada.
        public_key_path (str): Ruta para guardar la clave pública.
        passphrase (str, optional): Contraseña para cifrar la clave privada. Si es None, no se cifrará.
                                     ¡Recomendado para producción!
    """
    logger.info("Iniciando la generación de claves RSA para ERP.")

    # 1. Generar la clave privada RSA
    # Se recomienda una longitud de clave de al menos 2048 bits, 4096 es aún más seguro.
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048, # Puedes cambiar a 4096 para mayor seguridad si lo deseas
        backend=default_backend()
    )
    logger.info("Clave privada RSA generada en memoria.")

    # 2. Obtener la clave pública de la clave privada
    public_key = private_key.public_key()
    logger.info("Clave pública extraída de la clave privada.")

    # 3. Serializar y guardar la clave privada
    # La clave privada se guarda en formato PEM
    # Se recomienda usar una contraseña (passphrase) para mayor seguridad
    encryption_algorithm = serialization.NoEncryption()
    if passphrase:
        encryption_algorithm = serialization.BestAvailableEncryption(passphrase.encode())
        logger.info("Se usará una contraseña para cifrar la clave privada.")
    else:
        logger.warning("La clave privada NO se cifrará con contraseña. ¡Esto no es seguro para entornos de producción!")

    try:
        with open(private_key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.OpenSSH, # Formato específico de OpenSSH para privada
                encryption_algorithm=encryption_algorithm
            ))
        logger.info(f"Clave privada guardada en: {private_key_path}")
    except IOError as e:
        logger.error(f"Error al guardar la clave privada: {e}")
        return False

    # 4. Serializar y guardar la clave pública
    # La clave pública se guarda en formato OpenSSH, el más común para 'authorized_keys'
    try:
        with open(public_key_path, "wb") as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.OpenSSH,
                format=serialization.PublicFormat.OpenSSH
            ))
        logger.info(f"Clave pública guardada en: {public_key_path}")
    except IOError as e:
        logger.error(f"Error al guardar la clave pública: {e}")
        return False

    logger.info("Generación de claves completada exitosamente.")
    return True

if __name__ == "__main__":
    # --- Ejecución del script ---
    # Puedes solicitar una contraseña al usuario o dejarla como None para pruebas.
    # Para fines de este examen, puedes dejarla como None para simplificar,
    # pero entiende la implicación de seguridad.
    
    # Ejemplo con contraseña (descomentar y modificar para usarla)
    # user_passphrase = input("Introduce una contraseña para la clave privada (dejar vacío para ninguna): ")
    # if user_passphrase == "":
    #     user_passphrase = None
    # if generate_ssh_keys(passphrase=user_passphrase):

    # Ejemplo sin contraseña (más fácil para pruebas iniciales)
    if generate_ssh_keys():
        logger.info("Proceso de generación de claves finalizado.")
    else:
        logger.error("La generación de claves encontró problemas.")