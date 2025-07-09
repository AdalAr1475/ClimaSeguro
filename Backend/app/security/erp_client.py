import paramiko
import time
import json
import os
import socket

# Importar el gestor de logs
from log_manager import setup_logger

logger = setup_logger(__name__)

# --- Configuración del Cliente ---
SERVER_HOST = '127.0.0.1' # La IP o hostname donde tu ssh_server.py está corriendo
SERVER_PORT = 2222        # El puerto donde tu ssh_server.py está escuchando
SSH_USERNAME = 'erp_user' # El nombre de usuario que el servidor espera para la autenticación por clave
PRIVATE_KEY_PATH = "id_rsa_erp" # Ruta a la clave privada generada por keygen.py
# Si tu clave privada tiene contraseña, descomenta la siguiente línea y reemplaza con la contraseña
# PRIVATE_KEY_PASSPHRASE = "tu_contraseña_aqui"


def get_sensor_data_from_ssh():
    """
    Se conecta al servidor SSH, autentica con clave privada,
    y solicita datos del sensor.
    """
    logger.info(f"Intentando conectar al servidor SSH en {SERVER_HOST}:{SERVER_PORT} como '{SSH_USERNAME}'...")
    
    client = paramiko.SSHClient()
    
    # 1. Configurar política de claves de host (para evitar errores al desconocer el host)
    # AutoAddPolicy es conveniente para pruebas, en producción usar RejectPolicy o WarningPolicy
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 2. Cargar la clave privada del ERP
    try:
        # Si la clave privada tiene contraseña, descomenta la línea con 'password'
        private_key = paramiko.RSAKey.from_private_key_file(
            PRIVATE_KEY_PATH,
            # password=PRIVATE_KEY_PASSPHRASE # Descomentar si tu clave privada tiene contraseña
        )
        logger.info(f"Clave privada cargada desde: {PRIVATE_KEY_PATH}")
    except paramiko.ssh_exception.SSHException as e:
        logger.error(f"Error al cargar la clave privada: {e}")
        return None
    except FileNotFoundError:
        logger.error(f"Error: Clave privada no encontrada en {PRIVATE_KEY_PATH}. Asegúrate de ejecutar keygen.py.")
        return None
    except Exception as e:
        logger.error(f"Error inesperado al cargar la clave privada: {e}")
        return None

    # 3. Establecer la conexión SSH
    try:
        client.connect(
            hostname=SERVER_HOST,
            port=SERVER_PORT,
            username=SSH_USERNAME,
            pkey=private_key,
            timeout=10 # Timeout para la conexión
        )
        logger.info("Conexión SSH establecida exitosamente.")

        # 4. Ejecutar comando remoto para obtener datos del sensor
        command = "GET_SENSOR_DATA" # Este comando debe ser reconocido por ssh_server.py
        logger.info(f"Enviando comando: '{command}' al servidor.")
        stdin, stdout, stderr = client.exec_command(command)

        # 5. Leer la salida (datos del sensor)
        output = stdout.read().decode('utf-8').strip()
        error = stderr.read().decode('utf-8').strip()

        if output:
            logger.info("Datos recibidos del servidor:")
            logger.info(output)
            try:
                sensor_data = json.loads(output)
                logger.info("Datos del sensor parseados como JSON.")
                return sensor_data
            except json.JSONDecodeError as e:
                logger.error(f"Error al parsear la respuesta JSON: {e}")
                logger.error(f"Respuesta cruda: {output}")
                return None
        elif error:
            logger.error(f"Error recibido del servidor: {error}")
            return None
        else:
            logger.warning("El servidor no devolvió datos ni errores para el comando.")
            return None

    except paramiko.AuthenticationException:
        logger.error("Fallo de autenticación. Verifica el usuario y la clave privada.")
        return None
    except paramiko.SSHException as e:
        logger.error(f"Error SSH al conectar o ejecutar comando: {e}")
        return None
    except socket.error as e:
        logger.error(f"Error de socket (conexión rechazada/no disponible): {e}")
        logger.error(f"Asegúrate de que ssh_server.py esté ejecutándose en {SERVER_HOST}:{SERVER_PORT}.")
        return None
    except Exception as e:
        logger.error(f"Error inesperado durante la conexión SSH: {e}", exc_info=True)
        return None
    finally:
        if client:
            client.close()
            logger.info("Conexión SSH cerrada.")

if __name__ == "__main__":
    logger.info("Iniciando cliente ERP para obtener datos del sensor.")
    
    # Asegúrate de que la clave privada exista antes de intentar conectar
    if not os.path.exists(PRIVATE_KEY_PATH):
        logger.error(f"El archivo de clave privada '{PRIVATE_KEY_PATH}' no existe. Por favor, ejecuta 'python keygen.py' primero.")
    else:
        # Puedes llamar a la función en un bucle para simular lecturas periódicas
        # Aunque para esta prueba inicial, una sola llamada es suficiente.
        sensor_reading = get_sensor_data_from_ssh()

        if sensor_reading:
            logger.info(f"ERP recibió y procesó datos del sensor: {sensor_reading}")
            # Aquí podrías integrar la lógica para enviar al "banco" o procesar
        else:
            logger.error("ERP no pudo obtener los datos del sensor.")

    logger.info("Cliente ERP finalizado.")