# Лабораторная работа №1 по дисциплине Модели решения задач в интеллектуальных системах.
# Вариант 7. Реализовать алгоритм вычисления произведения пары 6-разрядных чисел умножением с младших разрядов со сдвигом множимого влево
# Выполнена студентом группы 221702 БГУИР Багдасаров Иван Евгеньевич
# Программа выполняет умножение пары чисел.

def main():
   init = init_vectors()
   A, B = [], []
   if init is None:
       return
   A, B = init

def init_vectors():
    print("Enter А: ")
    first = input()
    print("Enter В: ")
    second = input()

    A = [int(num.strip()) for num in first.split(',') if num.strip().isdigit()]
    B = [int(num.strip()) for num in second.split(',') if num.strip().isdigit()]

    if len(A) != len(B):
        print("Unacceptable values")
        return None

    check = True
    for i in range(len(A)):
        if not values_allowed(A[i], B[i]):
            check = False
            break

    if not check:
        print("Unacceptable values")
        return None
    return A, B

def values_allowed(a, b):
    if a > 63 or b > 63:
        return False
    return True

def add_binary(bin1, bin2):
    max_len = max(len(bin1), len(bin2))
    bin1 = bin1.zfill(max_len)
    bin2 = bin2.zfill(max_len)

    carry = 0
    result = []

    for i in range(max_len - 1, -1, -1):
        total = carry
        total += 1 if bin1[i] == '1' else 0
        total += 1 if bin2[i] == '1' else 0

        result.append('1' if total % 2 == 1 else '0')
        carry = total // 2

    if carry:
        result.append('1')

    result.reverse()
    return ''.join(result).zfill(12)[-12:]


def shift_binary(bin1):
    shifted = bin1 + '0'
    return shifted.zfill(12)[-12:]




