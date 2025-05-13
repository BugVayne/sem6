# Лабораторная работа №1 по дисциплине Модели решения задач в интеллектуальных системах.
# Вариант 7. Реализовать алгоритм вычисления произведения пары 6-разрядных чисел умножением с младших разрядов со сдвигом множимого влево.
# Выполнена студентом группы 221702 БГУИР Багдасаров Иван Евгеньевич





import sys
import os

def validate_number_range(num1, num2):
    return num1 <= 63 and num2 <= 63


def binary_sum(binary1: str, binary2: str) -> str:
    """Суммирует два бинарных числа и возвращает результат в 12-битном формате."""
    # Make both binary strings the same length by padding with zeros
    max_len = max(len(binary1), len(binary2))
    binary1 = binary1.zfill(max_len)
    binary2 = binary2.zfill(max_len)

    carry = 0
    result = []

    # Perform binary addition from right to left
    for b1, b2 in zip(reversed(binary1), reversed(binary2)):
        total = carry + (b1 == '1') + (b2 == '1')
        result_bit = total % 2  # Result bit
        carry = total // 2       # Carry for the next bit
        result.append(str(result_bit))

    # If there's a carry left, append it
    if carry:
        result.append('1')

    # Reverse the result and join to form the final binary string
    result.reverse()
    binary_result = ''.join(result)

    # Return the result in 12-bit format
    return binary_result.zfill(12)


def calculate_partial_product(state, flag = True):
    """Обрабатывает текущий этап алгоритма умножения."""
    multiplicand = state['multiplicand']
    multiplier = state['multiplier']
    partial_product = state['partial_product']
    partial_sum = state['partial_sum']


    if flag:
        partial_product = partial_product[1:] + '0'
    if multiplier[-1] == '1':  # Проверяем младший разряд множителя
        partial_sum = binary_sum(partial_sum, partial_product)  # Суммируем с текущей частичной суммой

    return {
        'multiplicand': multiplicand,
        'multiplier': multiplier,
        'partial_product': partial_product,
        'partial_sum': partial_sum
    }

def display_step(queue, pipeline_stages, results, cycle):
    """Выводит текущее состояние алгоритма на каждом такте."""
    print(f"Такт {cycle + 1}")
    print("Входная очередь:")
    if queue:
        for pair in queue:
            print(f"{pair['multiplicand']} и {pair['multiplier']}")
    else:
        print("-")

    for stage_index in range(6):
        print(f"Этап {stage_index + 1}")
        if pipeline_stages[stage_index]:
            stage = pipeline_stages[stage_index]
            print(f"Множимое: {stage['multiplicand']}")
            print(f"Множитель: {stage['multiplier']}")
            print(f"Частичное произведение: {stage['partial_product']}")
            print(f"Частичная сумма: {stage['partial_sum']}")
        else:
            print("-")

    if results:
        print("Результат:")
        for result in results:
            print(f"{result} ({int(result, 2)})")
    else:
        print("-")

    print("1. Дальше")
    print("2. Выход")
    user_choice = input()
    if user_choice == '1':
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        sys.exit()

def main():
    while True:
        print("Введите вектор A (через запятую): ")
        input_a = input()
        print("Введите вектор B (через запятую): ")
        input_b = input()

        vector_a = [int(num.strip()) for num in input_a.split(',') if num.strip().isdigit()]
        vector_b = [int(num.strip()) for num in input_b.split(',') if num.strip().isdigit()]

        if len(vector_a) != len(vector_b):
            print("Размеры векторов не совпадают!")
            return

        if not all(validate_number_range(a, b) for a, b in zip(vector_a, vector_b)):
            print("Одно из чисел превышает 6 разрядов!")
            return

        # Инициализация
        total_cycles = 6 + len(vector_a) - 1
        input_queue = [{'multiplicand': a, 'multiplier': b} for a, b in zip(vector_a, vector_b)]
        pipeline_stages = [None] * 6
        results = []

        # Подготовка данных
        for pair in input_queue:
            pair['multiplicand'] = bin(pair['multiplicand'])[2:].zfill(6)  # Преобразуем множимое в двоичное
            pair['multiplier'] = bin(pair['multiplier'])[2:].zfill(6)  # Преобразуем множитель в двоичное
            pair['partial_product'] = '0'*6 +pair['multiplicand']  # Начальное частичное произведение
            pair['partial_sum'] = '0' * 12  # Изначальная частичная сумма равна 0

        # Запуск тактов
        display_step(input_queue, pipeline_stages, results, -1)  # Показать начальное состояние
        for cycle in range(total_cycles):
            for stage_index in range(5, -1, -1):  # Проход по этапам от последнего к первому
                if stage_index == 0 and input_queue:  # Первый этап обрабатывает данные из очереди
                    pipeline_stages[0] = calculate_partial_product(input_queue.pop(0), flag=False)
                elif pipeline_stages[stage_index - 1] and stage_index>0:  # Обрабатываем данные из предыдущего этапа
                    current_stage = pipeline_stages[stage_index - 1]
                    current_stage['multiplier'] = '0' + current_stage['multiplier'][:-1]
                    pipeline_stages[stage_index] = calculate_partial_product(current_stage)
                    pipeline_stages[stage_index - 1] = None  # Освобождаем предыдущий этап
                    if stage_index == 5:  # Если этап последний, сохраняем результат
                        results.append(pipeline_stages[stage_index]['partial_sum'])

            display_step(input_queue, pipeline_stages, results, cycle)

        # Завершающая печать
        display_step([], [None] * 6, results, -1)


if __name__ == "__main__":
    main()