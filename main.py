import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, Listbox, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from decimal import Decimal,  getcontext, ROUND_DOWN

def setup_style():
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', background='#E1E1E1', font=('Helvetica', 10))
    style.configure('TLabel', font=('Helvetica', 10))
    style.configure('TEntry', font=('Helvetica', 10))

def euclidean_distance(vec1, vec2):
    getcontext().prec = 10
    vec1 = [float(x) if x is not None else np.nan for x in vec1]
    vec2 = [float(x) if x is not None else np.nan for x in vec2]
    return np.nansum((np.array(vec1) - np.array(vec2))**2)**0.5

def plot_similarity(similarity_pairs):
    fig, ax = plt.subplots()
    similarities, labels = zip(*similarity_pairs)
    bars = ax.bar(labels, similarities, color='skyblue', edgecolor='black')
    ax.set_xlabel('Пары оборудования')
    ax.set_ylabel('Процент сходства')
    ax.set_title('Процент сходства на основе евклидовых расстояний')
    ax.set_ylim(0, 100)
    for bar, value in zip(bars, similarities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.1f}%', ha='center', va='bottom')
    return fig, ax

def update_listbox():
    listbox.delete(0, tk.END)
    for name, vector in zip(names, vectors):
        listbox.insert(tk.END, f"{name}: {vector}")

def add_vector():
    name = simpledialog.askstring("Введите название оборудования", "Название оборудования:")
    if not name:
        messagebox.showerror("Ошибка", "Необходимо ввести название оборудования.")
        return
    num_features = simpledialog.askinteger("Количество признаков", "Введите количество признаков вектора:")
    if not num_features or num_features < 1:
        messagebox.showerror("Ошибка", "Некорректное количество признаков.")
        return
    vector = []
    for i in range(num_features):
        feature_value = simpledialog.askstring(f"Признак {i + 1}", f"Введите значение для признака {i + 1} (оставьте пустым для None):")
        if feature_value == '':
            vector.append(None)
        else:
            try:
                vector.append(float(feature_value))
            except ValueError:
                messagebox.showerror("Ошибка", f"Признак {i + 1} должен быть числом или пустым.")
                return
    vectors.append(vector)
    names.append(name)
    update_listbox()

def delete_vector():
    selected_indices = listbox.curselection()
    if not selected_indices:
        messagebox.showwarning("Предупреждение", "Выберите вектор(ы) для удаления.")
        return
    for index in sorted(selected_indices, reverse=True):
        del vectors[index]
        del names[index]
    update_listbox()

def edit_vector():
    selected_index = listbox.curselection()
    if not selected_index:
        messagebox.showwarning("Предупреждение", "Выберите вектор для редактирования.")
        return
    selected_index = selected_index[0]
    new_vector_str = simpledialog.askstring("Редактирование вектора", "Введите новый вектор через запятую:")
    if new_vector_str is None:
        return
    try:
        new_vector = [float(x.strip()) if x.strip() != '' else None for x in new_vector_str.split(',')]
        vectors[selected_index] = new_vector
        update_listbox()
    except ValueError:
        messagebox.showerror("Ошибка", "Вектор должен содержать только числа.")

def calculate_similarity():
    global similarity_pairs, canvas
    getcontext().prec = 28  # Установка высокой точности для Decimal
    if len(vectors) < 2:
        messagebox.showinfo("Информация", "Необходимо добавить минимум два вектора.")
        return

    distance_pairs = [(Decimal(euclidean_distance(vectors[i], vectors[j])), f'{names[i]}-{names[j]}')
                      for i in range(len(vectors)) for j in range(i + 1, len(vectors))]

    distances = [pair[0] for pair in distance_pairs]
    mean_dist = Decimal(np.mean([float(d) for d in distances]))  # Преобразование в float, затем обратно в Decimal
    std_dist = Decimal(np.std([float(d) for d in distances]))

    similarity_percentages = [100 * (1 - Decimal(norm.cdf(float((dist - mean_dist) / std_dist)))) for dist in distances]
    similarity_pairs = list(zip(similarity_percentages, [pair[1] for pair in distance_pairs]))

    if canvas:
        canvas.get_tk_widget().pack_forget()
    fig, ax = plot_similarity(similarity_pairs)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def save_similarity_results():
    if not similarity_pairs:
        messagebox.showerror("Ошибка", "Нет данных для сохранения. Сначала рассчитайте подобие.")
        return
    date_similarity = simpledialog.askstring("Дата проведения анализа", "Введите дату проведения анализа в формате dd:mm:yy:")
    equipment_name = simpledialog.askstring("Название структурного элемента", "Введите название структурного элемента оборудования:")
    if not equipment_name:
        messagebox.showerror("Ошибка", "Необходимо ввести название структурного элемента.")
        return
    filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if not filepath:
        return
    with open(filepath, 'w') as file:
        file.write(f"Дата проведения анализа: {date_similarity}\n")
        file.write(f"Структурный элемент оборудования: {equipment_name}\n")
        for percent, pair in similarity_pairs:
            file.write(f'{pair}: {percent:.2f}%\n')
    messagebox.showinfo("Сохранение", "Данные успешно сохранены!")

def clear_plot():
    global canvas
    if canvas:
        canvas.get_tk_widget().pack_forget()


def standardize_key(key):
    replacements = {'Т': 'T', 'Н': 'N', 'Е': 'E', 'С': 'C'}
    return ''.join(replacements.get(char, char) for char in key)


def process_similarity_results():
    global similarity_aggregate
    highest_similarity = 0
    highest_similarity_equipment = None

    for pair, values in similarity_aggregate.items():
        if len(values) >= 4:
            average_percent = np.mean(values)
            if average_percent > highest_similarity and average_percent >= 90:
                highest_similarity = average_percent
                highest_similarity_equipment = pair

    if highest_similarity_equipment:
        similarity_coefficient = highest_similarity / 100
        messagebox.showinfo("Наиболее похожее оборудование",
                            f"Оборудование {highest_similarity_equipment} имеет высокий процент сходства: {highest_similarity:.2f}%")
        request_failure_rate_input()

def request_failure_rate_input():
    global highest_similarity_equipment, similarity_coefficient
    if not highest_similarity_equipment:
        return
    failure_rate = simpledialog.askfloat("Интенсивность отказов",
                                         "Введите интенсивность отказов для оборудования-прототипа (1/ч):",
                                         minvalue=0.0)
    if failure_rate is None:
        return

    calculated_failure_rate = failure_rate * similarity_coefficient
    messagebox.showinfo("Расчетная интенсивность отказов",
                        f"Расчетная интенсивность отказов для {highest_similarity_equipment} составляет: {calculated_failure_rate:.6f} 1/ч")


similarity_aggregate = {}
def load_similarity_results():
    global similarity_aggregate

    file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt"), ("All files", "*.*")], initialdir="/")
    if not file_paths:
        return

    if not hasattr(load_similarity_results, 'similarity_aggregate'):
        load_similarity_results.similarity_aggregate = {}

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                if ':' in line:
                    try:
                        pair, percent = line.strip().split(':')
                        pair = standardize_key(pair.strip())  
                        percent = float(percent.replace('%', ''))
                        if pair not in load_similarity_results.similarity_aggregate:
                            load_similarity_results.similarity_aggregate[pair] = []
                        load_similarity_results.similarity_aggregate[pair].append(percent)

                        print(f"Updated pair: {pair}")
                        print(f"Values for {pair}: {load_similarity_results.similarity_aggregate[pair]}")
                    except ValueError:
                        continue

    averaged_similarity = []
    for pair, values in load_similarity_results.similarity_aggregate.items():
        average_percent = np.mean(values)
        averaged_similarity.append((average_percent, pair))
        print(f"Average for {pair}: {average_percent}%")

    global similarity_pairs
    similarity_pairs = sorted(averaged_similarity,
                              key=lambda x: x[1])

    plot_and_display_similarity()
    process_similarity_results()

def plot_and_display_similarity():
    global canvas, similarity_pairs
    if similarity_pairs:  
        if canvas:
            canvas.get_tk_widget().pack_forget()
        fig, ax = plot_similarity(similarity_pairs)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    else:
        messagebox.showerror("Ошибка", "Нет данных подобия для отображения.")

def calculate_reliability():
    def perform_calculation():
        getcontext().prec = 50  
        decimal_format = Decimal('0.00000001') 

        try:
            failure_rate = Decimal(failure_rate_entry.get())
            recovery_rate = Decimal(recovery_rate_entry.get())
            reliability_coefficient = Decimal(reliability_coefficient_entry.get())
            is_more_complex = complexity_var.get()

            if failure_rate <= 0 or reliability_coefficient <= 0 or recovery_rate <= 0:
                messagebox.showerror("Ошибка", "Значения должны быть больше нуля.")
                return

            calculated_failure_rate = failure_rate / reliability_coefficient if is_more_complex == 'Сложнее' else failure_rate * reliability_coefficient
            calculated_recovery_rate = recovery_rate * reliability_coefficient if is_more_complex == 'Сложнее' else recovery_rate/reliability_coefficient
            mean_time_to_failure = Decimal('1') / calculated_failure_rate if calculated_failure_rate != 0 else Decimal('inf')
            recovery_time = Decimal('1') / calculated_recovery_rate if recovery_rate != 0 else Decimal('inf')
            readiness_coefficient = mean_time_to_failure / (mean_time_to_failure + recovery_time)

            # Округляем значения до 8 знаков после запятой без округления вверх
            formatted_failure_rate = calculated_failure_rate.quantize(decimal_format, rounding=ROUND_DOWN)
            formatted_recovery_rate = calculated_recovery_rate.quantize(decimal_format, rounding=ROUND_DOWN)
            formatted_mttf = mean_time_to_failure.quantize(decimal_format, rounding=ROUND_DOWN)
            formatted_recovery_time = recovery_time.quantize(decimal_format, rounding=ROUND_DOWN)
            formatted_readiness = readiness_coefficient.quantize(decimal_format, rounding=ROUND_DOWN)

            result_text = (f"Рассчитанная интенсивность отказов: {formatted_failure_rate}\n"
                           f"Рассчитанная интенсивность восстановления: {formatted_recovery_rate}\n"
                           f"Время наработки на отказ: {formatted_mttf} часов\n"
                           f"Время восстановления: {formatted_recovery_time} часов\n"
                           f"Коэффициент готовности: {formatted_readiness}")
            print(f"Результаты расчёта: {result_text}")
            messagebox.showinfo("Результаты расчёта", result_text)
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректный ввод данных. Убедитесь, что введены числа.\nОшибка: {e}")
        except Exception as e:
            messagebox.showerror("Неизвестная ошибка", f"Произошла неожиданная ошибка: {e}")

    reliability_window = tk.Toplevel(root)
    reliability_window.title("Расчет показателей надежности")
    reliability_window.geometry("500x400")

    tk.Label(reliability_window, text="Интенсивность отказов (1/ч):").pack(pady=10)
    failure_rate_entry = tk.Entry(reliability_window)
    failure_rate_entry.pack(pady=5)

    tk.Label(reliability_window, text="Интенсивность восстановления (1/ч):").pack(pady=10)
    recovery_rate_entry = tk.Entry(reliability_window)
    recovery_rate_entry.pack(pady=5)

    tk.Label(reliability_window, text="Коэффициент надежности:").pack(pady=10)
    reliability_coefficient_entry = tk.Entry(reliability_window)
    reliability_coefficient_entry.pack(pady=5)

    tk.Label(reliability_window, text="Оцените сложность нового изделия относительно его прототипа:").pack(pady=10)
    complexity_var = tk.StringVar(value='Сложнее')
    tk.Radiobutton(reliability_window, text='Сложнее', variable=complexity_var, value='Сложнее').pack()
    tk.Radiobutton(reliability_window, text='Проще', variable=complexity_var, value='Проще').pack()

    tk.Button(reliability_window, text="Рассчитать", command=perform_calculation).pack(pady=20)

def main():
    global root, listbox, canvas, vectors, names, similarity_pairs
    similarity_pairs = []
    root = tk.Tk()
    root.title("Анализ подобия оборудования")
    root.geometry('1500x700')
    setup_style()
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    listbox = Listbox(main_frame, height=6)
    listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
    buttons_frame = ttk.Frame(main_frame)
    buttons_frame.pack(fill=tk.X, expand=True)

    ttk.Button(buttons_frame, text="Добавить вектор", command=add_vector).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Удалить выбранный вектор(ы)", command=delete_vector).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Редактировать вектор", command=edit_vector).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Рассчитать подобие оборудования", command=calculate_similarity).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Сохранить значения подобия", command=save_similarity_results).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Загрузить результаты", command=load_similarity_results).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Очистить график", command=clear_plot).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Рассчитать показатели надежности", command=calculate_reliability).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5,  pady=5)

    canvas = None
    root.mainloop()

if __name__ == "__main__":
    main()

