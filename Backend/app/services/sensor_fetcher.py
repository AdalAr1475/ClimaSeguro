# app/services/sensor_fetcher.py
import paramiko
import json
from datetime import datetime
import os # Importar os para la ruta absoluta

# Asumiendo que 'log_manager.py' está disponible en el directorio raíz o en una ruta accesible
from ..security.log_manager import setup_logger 

logger = setup_logger(__name__)

# --- Configuración del servidor SSH (debe coincidir con ssh_server.py) ---
SSH_HOST = '0.0.0.0' # O la IP donde esté corriendo tu ssh_server.py
SSH_PORT = 2222
SSH_USERNAME = 'testuser' # Debe coincidir con el usuario configurado en CustomSSHServer

# --- NUEVO CÁLCULO DE LA RUTA DE LA CLAVE PRIVADA ---
# Obtener la ruta absoluta del directorio actual de este archivo (sensor_fetcher.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Construir la ruta a la raíz del proyecto (subiendo dos niveles desde app/services/)
PROJECT_ROOT = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
# Definir la ruta completa a la clave privada del cliente
PRIVATE_KEY_PATH = os.path.join(PROJECT_ROOT, 'id_rsa_erp') 
# --- FIN DE CÁLCULO DE LA RUTA ---


async def fetch_sensor_data_via_ssh():
    """
    Se conecta al servidor SSH, ejecuta el comando para obtener datos del sensor,
    y retorna los datos parseados.
    """
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    # Solo para desarrollo y pruebas: AutoAddPolicy añade automáticamente claves de host desconocidas.
    # EN PRODUCCIÓN: Usa RejectPolicy o una lista de HostKeys conocidas para mayor seguridad.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 

    try:
        private_key = paramiko.RSAKey.from_private_key_file(PRIVATE_KEY_PATH)
        logger.info(f"Intentando conectar a SSH en {SSH_HOST}:{SSH_PORT} como {SSH_USERNAME} con clave {PRIVATE_KEY_PATH}...")
        client.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USERNAME, pkey=private_key)
        logger.info("Conexión SSH establecida exitosamente.")

        # Ejecutar el comando para obtener los datos del sensor
        stdin, stdout, stderr = client.exec_command("GET_SENSOR_DATA")
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()

        if error:
            logger.error(f"Error al ejecutar comando SSH 'GET_SENSOR_DATA': {error}")
            return None

        logger.info(f"Datos raw recibidos vía SSH: {output}")

        try:
            sensor_data = json.loads(output)
            # Asegurarse de que el timestamp sea un objeto datetime para la validación de Pydantic
            if isinstance(sensor_data.get('timestamp'), str):
                sensor_data['timestamp'] = datetime.fromisoformat(sensor_data['timestamp'])
            
            logger.info("Datos del sensor parseados exitosamente.")
            return sensor_data
        except json.JSONDecodeError as e:
            logger.error(f"Error al parsear JSON de datos del sensor. Datos raw: '{output}'. Error: {e}")
            return None

    except FileNotFoundError:
        logger.error(f"Error: Clave privada SSH no encontrada en: {PRIVATE_KEY_PATH}. Asegúrate de que 'id_rsa_erp' exista y sea accesible.")
        return None
    except paramiko.AuthenticationException:
        logger.error("Fallo de autenticación SSH. Verifica el usuario, la clave privada y que la clave pública esté en el servidor SSH.")
        return None
    except paramiko.SSHException as e:
        logger.error(f"Error SSH (conexión o ejecución): {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado al intentar obtener datos del sensor vía SSH: {e}", exc_info=True)
        return None
    finally:
        if client:
            client.close()
            logger.info("Conexión SSH cerrada.")