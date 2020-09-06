import matplotlib.pyplot as plt


def sample_1():

    # Создание объекта Figure
    fig = plt.figure()
    # Список текущих областей рисования пуст
    print(fig.axes)
    # тип объекта Figure
    print(type(fig))
    # scatter - метод для нанесения маркера в точке (1.0, 1.0)
    plt.scatter(1.0, 1.0)
    # После нанесения графического элемента в виде маркера
    # список текущих областей состоит из одной области
    print(fig.axes)
    plt.show()


if __name__ == "__main__":
    sample_1()
