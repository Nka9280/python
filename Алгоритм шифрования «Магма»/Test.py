import struct
import os

# S-блоки (таблицы замен)
S_boxes = [
    [1, 7, 14, 13, 0, 5, 8, 3, 4, 15, 10, 6, 9, 12, 11, 2],
    [8, 14, 2, 5, 6, 9, 1, 12, 15, 4, 11, 0, 13, 10, 3, 7],
    [5, 13, 15, 6, 9, 2, 12, 10, 11, 7, 8, 1, 4, 3, 14, 0],
    [7, 15, 5, 10, 8, 1, 6, 13, 0, 9, 3, 14, 11, 4, 2, 12],
    [12, 8, 2, 1, 13, 4, 15, 6, 7, 0, 10, 5, 3, 14, 9, 11],
    [11, 3, 5, 8, 2, 15, 10, 13, 14, 1, 7, 4, 12, 9, 6, 0],
    [6, 8, 2, 3, 9, 10, 5, 12, 1, 14, 4, 7, 11, 13, 0, 15],
    [12, 4, 6, 2, 10, 5, 11, 9, 14, 8, 13, 7, 0, 3, 15, 1]
]

# Функция для шифрования одного 64-битного блока с использованием алгоритма "Магма"
def encrypt_block(block, keys):
    left = block >> 32
    right = block & 0xFFFFFFFF

    for round in range(32):
        key_index = round % 16
        if round < 24:
            key_index = round % 8
        else:
            key_index = 7 - (round % 8)

        # Добавляем ключ и применяем S-блоки
        right = (right + keys[key_index]) % 0x100000000
        s_input = [(right >> (i * 4)) & 0xF for i in range(8)]
        s_output = 0

        # Применение S-блоков к каждому 4-битному фрагменту
        for i in range(8):
            s_output |= S_boxes[i][s_input[i]] << (i * 4)

        # Вращение на 11 битов влево
        s_output = ((s_output << 11) | (s_output >> (32 - 11))) & 0xFFFFFFFF

        # Меняем местами левую и правую части
        if round == 31:
            left = s_output
        else:
            new_left = left ^ s_output
            left, right = right, new_left

    return (left << 32) | right

# Функция для расшифрования блока (обратный процесс, идентичен для OFB)
def decrypt_block(block, keys):
    return encrypt_block(block, keys)

# Функция гаммирования (режим OFB)
def ofb_mode(input_filepath, output_filepath, key, iv, is_decryption=False, original_size=None):
    # Генерация 256-битных ключей (8 блоков по 32 бита)
    keys = [struct.unpack('<I', key[i:i + 4])[0] for i in range(0, 32, 4)]
    gamma = struct.unpack('<Q', iv)[0]  # Начальный вектор IV преобразуем в 64-битное число
    
    with open(input_filepath, 'rb') as f_in, open(output_filepath, 'wb') as f_out:
        total_read = 0  # Общее количество прочитанных байт
        
        while True:
            block = f_in.read(8)  # Чтение 64-битных блоков (8 байт)
            if not block:
                break

            total_read += len(block)
            # Генерация гаммы с помощью шифрования предыдущего значения гаммы
            gamma = encrypt_block(gamma, keys)

            # Преобразование блока данных в целое число
            block_int = struct.unpack('<Q', block.ljust(8, b'\x00'))[0]

            # XOR блока данных с текущей гаммой
            output_block_int = block_int ^ gamma

            # Запись зашифрованного/расшифрованного блока в файл
            f_out.write(struct.pack('<Q', output_block_int))

        # Если это расшифрование, удалим добавленные нули
        if is_decryption and original_size is not None:
            f_out.truncate(original_size)  # Обрезка файла до оригинального размера

# Пример использования программы
def main():
    input_file = 'input.txt'  # Замените на ваш файл
    encrypted_file = 'encrypted.bin'
    decrypted_file = 'decrypted.txt'

    # Пример 256-битного секретного ключа и 64-битного IV
    secret_key = b'06309328904834842791716872210665'  # Должно быть 32 байта
    iv = b'12345678'  # Длина вектора инициализации: 8 байт

    # Получаем размер исходного файла
    original_size = os.path.getsize(input_file)

    # Шифрование файла
    print("Шифрование файла...")
    ofb_mode(input_file, encrypted_file, secret_key, iv)

    # Дешифрование файла
    print("Дешифрование файла...")
    ofb_mode(encrypted_file, decrypted_file, secret_key, iv, is_decryption=True, original_size=original_size)

if __name__ == "__main__":
    main()
