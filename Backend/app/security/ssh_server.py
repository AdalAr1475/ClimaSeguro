import socket
import threading
import paramiko
import os
import json
import time
import base64 # Importa base64 para decodificar

# Importar el gestor de logs
from .log_manager import setup_logger

logger = setup_logger(__name__)

# Importa tipos de clave específicos y excepciones de paramiko
from paramiko import RSAKey, DSSKey, ECDSAKey
from paramiko.ssh_exception import SSHException

# --- NUEVO CÁLCULO DE RUTAS ABSOLUTAS ---
# Obtener la ruta absoluta del directorio actual de este archivo (server.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Construir la ruta a la raíz del proyecto (subiendo dos niveles desde app/security/)
PROJECT_ROOT = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# --- Rutas de Archivos ACTUALIZADAS ---
# Clave privada del servidor (server_host_key.rsa estará en la raíz del proyecto)
HOST_KEY_PATH = os.path.join(PROJECT_ROOT, "server_host_key.rsa")
# Clave pública del cliente ERP para autenticación (id_rsa_erp.pub estará en la raíz del proyecto)
AUTHORIZED_KEYS_PATH = os.path.join(PROJECT_ROOT, "id_rsa_erp.pub")
# Archivo con la última lectura del sensor (AHORA APUNTA DENTRO DE app/security/ donde está server.py)
CURRENT_SENSOR_READING_PATH = os.path.join(current_script_dir, "current_sensor_reading.json") # <--- ¡LÍNEA MODIFICADA AQUÍ!

# --- FIN DE CÁLCULO DE RUTAS ---

# Función auxiliar para cargar una clave pública en formato OpenSSH desde una cadena
def _load_public_key_from_openssh_string(key_string):
    """
    Parsea una cadena de clave pública en formato OpenSSH (ej. 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC...')
    y retorna un objeto de clave paramiko (RSAKey, DSSKey, etc.).
    """
    parts = key_string.strip().split()
    if len(parts) < 2:
        raise SSHException("Formato de clave pública OpenSSH inválido: Faltan partes.")

    key_type = parts[0]
    key_data_b64 = parts[1]

    try:
        decoded_data = base64.b64decode(key_data_b64)
    except Exception as e:
        raise SSHException(f"Codificación base64 inválida en clave pública: {e}")

    try:
        msg = paramiko.message.Message(decoded_data)
        parsed_key_type = msg.get_text()
        
        if parsed_key_type == 'ssh-rsa':
            e = msg.get_mpint() # exponente público
            n = msg.get_mpint() # módulo

            new_msg_data = paramiko.message.Message()
            new_msg_data.add_string(b'ssh-rsa') # El tipo de clave
            new_msg_data.add_mpint(e)
            new_msg_data.add_mpint(n)
            return RSAKey(data=new_msg_data.asbytes())

        elif parsed_key_type == 'ssh-dss':
            p = msg.get_mpint()
            q = msg.get_mpint()
            g = msg.get_mpint()
            y = msg.get_mpint()
            return DSSKey(p=p, q=q, g=g, y=y)
        elif parsed_key_type.startswith('ecdsa-sha2-'):
            curve_name = msg.get_text()
            point_data = msg.get_string()
            return ECDSAKey(curve=curve_name, x=point_data)
        else:
            raise SSHException(f"Tipo de clave pública no soportado en blob: {parsed_key_type}")
    except Exception as e:
        raise SSHException(f"Error al parsear el blob de clave pública: {e}")

    
class CustomSSHServer(paramiko.ServerInterface):
    """
    Clase que implementa la interfaz de un servidor SSH para paramiko.
    """
    def __init__(self):
        self.event = threading.Event()
        self.public_key_to_authenticate = None
        self.load_authorized_keys()

    def load_authorized_keys(self):
        """Carga la clave pública del cliente ERP para la autenticación."""
        if not os.path.exists(AUTHORIZED_KEYS_PATH):
            logger.error(f"Archivo de clave pública del cliente no encontrado: {AUTHORIZED_KEYS_PATH}. "
                         "Por favor, genera las claves con keygen.py primero y asegúrate de que esté en la raíz del proyecto (ClimaSeguro/Backend/).")
            self.public_key_to_authenticate = None
            return

        try:
            with open(AUTHORIZED_KEYS_PATH, 'r') as f:
                public_key_content_line = f.read().strip() 
            
            self.public_key_to_authenticate = _load_public_key_from_openssh_string(public_key_content_line)
            
            logger.info(f"Clave pública del cliente ERP cargada desde {AUTHORIZED_KEYS_PATH}.")
        except Exception as e:
            logger.error(f"Error al cargar la clave pública del cliente ERP: {e}")
            logger.warning("Asegúrate de que la clave pública está en formato OpenSSH y sea compatible.")
            self.public_key_to_authenticate = None

    def check_auth_publickey(self, username, key):
        """
        Método para verificar la autenticación por clave pública.
        """
        logger.info(f"Intento de autenticación por clave pública para usuario: '{username}'")
        if self.public_key_to_authenticate is None:
            logger.warning("No hay clave pública del cliente cargada para verificar.")
            return paramiko.AUTH_FAILED

        # Compara la clave presentada por el cliente con la clave pública que tenemos cargada
        if username == 'testuser' and self.public_key_to_authenticate == key:
            logger.info(f"Autenticación exitosa para usuario: '{username}'")
            return paramiko.AUTH_SUCCESSFUL
        else:
            logger.warning(f"Autenticación fallida para usuario: '{username}'. Clave o usuario incorrecto.")
            return paramiko.AUTH_FAILED

    def get_allowed_auths(self, username):
        """
        Retorna los métodos de autenticación permitidos para el usuario.
        """
        logger.debug(f"Solicitud de métodos de autenticación permitidos para usuario: '{username}'")
        return 'publickey' # Solo permitimos autenticación por clave pública

    def check_channel_request(self, kind, chanid):
        """
        Verifica si el tipo de canal solicitado es permitido.
        """
        logger.info(f"Solicitud de canal: tipo='{kind}', id='{chanid}'")
        if kind == 'session':
            # Aceptar la solicitud de un canal de sesión
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_channel_shell_request(self, channel):
        """
        Verifica si se permite una shell interactiva (no necesaria para comandos simples).
        """
        logger.debug("Solicitud de shell interactiva. No permitida para este servidor simulado.")
        return False

    def check_channel_pty_request(self, channel, term, width, height, pixelwidth, pixelheight, modes):
        """
        Verifica si se permite una TTY (no necesaria para comandos simples).
        """
        logger.debug("Solicitud de TTY. No permitida para este servidor simulado.")
        return False
    
    def check_channel_exec_request(self, channel, command):
        """
        Maneja la ejecución de comandos remotos.
        """
        command_str = command.decode('utf-8').strip()
        logger.info(f"Comando recibido del cliente: '{command_str}'")

        if command_str == "GET_SENSOR_DATA":
            try:
                # Leer el archivo JSON con la última lectura del sensor
                if os.path.exists(CURRENT_SENSOR_READING_PATH):
                    with open(CURRENT_SENSOR_READING_PATH, 'r') as f:
                        sensor_data = json.load(f)
                    response = json.dumps(sensor_data, indent=2) # Formato legible
                    channel.sendall(response.encode('utf-8'))
                    channel.sendall(b"\n") # Asegurar nueva línea al final
                    logger.info("Datos del sensor enviados al cliente.")
                    channel.shutdown_write()
                    logger.debug("Extremo de escritura del canal cerrado para señalar fin de datos.")
                else:
                    error_msg = f"Error: Archivo de lectura de sensor no encontrado: {CURRENT_SENSOR_READING_PATH}"
                    channel.sendall(error_msg.encode('utf-8'))
                    logger.error(error_msg)
                    channel.shutdown_write()
            except json.JSONDecodeError as e:
                error_msg = f"Error al leer/parsear JSON del sensor: {e}"
                channel.sendall(error_msg.encode('utf-8'))
                logger.error(error_msg)
                channel.shutdown_write()
            except Exception as e:
                error_msg = f"Error inesperado al obtener datos del sensor: {e}"
                channel.sendall(error_msg.encode('utf-8'))
                logger.error(error_msg)
                channel.shutdown_write()
            return True
        else:
            logger.warning(f"Comando desconocido recibido: '{command_str}'")
            channel.sendall(f"Comando desconocido: {command_str}\n".encode('utf-8'))
            channel.shutdown_write()
            return True
    
# --- Función principal para iniciar el servidor SSH ---
def start_ssh_server(host='0.0.0.0', port=2222):
    """
    Inicia el servidor SSH simulado.
    """
    # 1. Generar o cargar la clave del host del servidor
    host_key = None
    if os.path.exists(HOST_KEY_PATH):
        try:
            host_key = paramiko.RSAKey.from_private_key_file(HOST_KEY_PATH)
            logger.info(f"Clave del host del servidor cargada desde {HOST_KEY_PATH}.")
        except paramiko.ssh_exception.SSHException as e:
            logger.error(f"Error al cargar la clave del host: {e}. Generando una nueva.")
            host_key = None
    
    if host_key is None:
        try:
            host_key = paramiko.RSAKey.generate(2048) # Generar una nueva clave RSA para el host
            host_key.write_private_key_file(HOST_KEY_PATH)
            logger.info(f"Nueva clave del host del servidor generada y guardada en {HOST_KEY_PATH}.")
        except Exception as e:
            logger.critical(f"Error FATAL al generar la clave del host: {e}. El servidor no puede iniciar.")
            return

    # 2. Configurar el socket del servidor
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False) # Hacer el socket no bloqueante
        sock.bind((host, port))
        sock.listen(1) # Escuchar una conexión a la vez
        logger.info(f"Servidor SSH simulado escuchando en {host}:{port}...")
    except Exception as e:
        logger.critical(f"Error al configurar el socket: {e}. El servidor no puede iniciar.")
        return

    # 3. Bucle para aceptar conexiones
    try:
        while True:
            conn = None
            try:
                conn, addr = sock.accept()
                logger.info(f"Conexión entrante de: {addr}")
            except BlockingIOError:
                time.sleep(0.1)
                continue
            except socket.timeout:
                    continue
            except Exception as e:
                logger.error(f"Error al aceptar conexión: {e}")
                if conn: conn.close()
                continue
            
            # 4. Iniciar el transporte SSH
            transport = paramiko.Transport(conn)
            transport.add_server_key(host_key)
            server = CustomSSHServer()
            try:
                transport.start_server(server=server)
                logger.info("Transporte SSH iniciado.")
            except paramiko.SSHException as e:
                logger.error(f"Error al iniciar el transporte SSH: {e}")
                transport.close()
                continue

            # 5. Esperar a que el canal se cierre (conexión terminada)
            channel = transport.accept(10) # Esperar hasta 10 segundos por un canal
            if channel is None:
                logger.warning("No se recibió ningún canal después de la conexión.")
                transport.close()
                continue
            else:
                logger.info("Canal de sesión aceptado.")
                while transport.is_active():
                    time.sleep(1)

            transport.close()
            logger.info(f"Conexión con {addr} cerrada.")

    except KeyboardInterrupt:
        logger.info("Servidor SSH simulado detenido por el usuario.")
    except Exception as e:
        logger.critical(f"Error inesperado en el bucle principal del servidor: {e}", exc_info=True)
    finally:
        if sock:
            sock.close()
        logger.info("Servidor SSH simulado finalizado.")

# --- Función asíncrona para ejecutar en FastAPI ---
async def start_ssh_server_async(host='0.0.0.0', port=2222):
    """
    Versión asíncrona del servidor SSH para ejecutar en el context manager de FastAPI.
    """
    import asyncio
    
    logger.info("Iniciando servidor SSH en modo asíncrono...")
    
    # Ejecutar la función bloqueante en un thread pool
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, start_ssh_server, host, port)

# --- Ejecución del script ---
if __name__ == "__main__":
    # Asegúrate de que el archivo de la clave pública del ERP exista
    if not os.path.exists(AUTHORIZED_KEYS_PATH):
        logger.warning(f"¡Advertencia! El archivo {AUTHORIZED_KEYS_PATH} no existe. Por favor, ejecuta 'python keygen.py' primero.")
        logger.warning("El servidor se iniciará, pero la autenticación por clave pública fallará sin el archivo.")
    
    start_ssh_server()