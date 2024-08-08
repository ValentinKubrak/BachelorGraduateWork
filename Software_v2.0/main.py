import torch
import flet as ft
import pathlib
from PIL import Image
import tempfile
import json
import os

# Перевизначення шляху для сумісності з Windows, оскільки pathlib.PosixPath може викликати проблеми на Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Завантаження файлу з мапуванням класів дорожніх знаків
with open('class_mapping.json', 'r') as f:
    class_mappings = json.load(f)

# Завантаження файлу з мапуванням значень дорожніх знаків
with open('meaning_mapping.json', 'r') as f:
    meaning_mappings = json.load(f)

# Завантаження моделі YOLOv5 з налаштуванням на користувацьку модель
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.iou = 0.45
model.conf = 0.25

def main(page: ft.Page):
    # Встановлення заголовку сторінки та налаштувань теми
    page.title = 'Traffic Sign Recognition App'
    page.theme_mode = 'light'
    page.window_resizable = False  # Заборона зміни розміру вікна

    # Створення контейнера для відображення завантаженого зображення (лівої сторони)
    global left_image_container
    left_image_container = ft.Container(
        width=400, height=400, bgcolor='lightgrey', border=ft.border.all(2, "blue"),
        alignment=ft.alignment.center, border_radius=ft.border_radius.all(20)
    )

    # Створення контейнера для відображення обробленого зображення (правої сторони)
    global right_image_container
    right_image_container = ft.Container(
        width=400, height=400, bgcolor='lightgrey', border=ft.border.all(2, "blue"),
        alignment=ft.alignment.center, border_radius=ft.border_radius.all(20)
    )

    # Створення об'єкта FilePicker для вибору файлів
    file_picker = ft.FilePicker(on_result=lambda picker_event: file_picker_result(picker_event))
    page.overlay.append(file_picker)  # Додавання FilePicker до оверлею сторінки

    # Контейнер для кнопок завантаження та обробки зображення
    global button_container
    button_container = ft.Container()

    # Глобальні змінні для зберігання шляхів до файлів та результатів
    global selected_file_path
    selected_file_path = None

    global processed_image_path
    processed_image_path = None

    global results
    results = None

    # Функція для завантаження фото через FilePicker
    def upload_photo(e):
        file_picker.pick_files(allow_multiple=False)  # Дозволяє вибрати лише один файл

    # Обробка результату вибору файлу
    def file_picker_result(picker_event):
        global selected_file_path, processed_image_path
        if picker_event.files:
            selected_file_path = picker_event.files[0].path  # Отримання шляху до вибраного файлу
            processed_image_path = None  # Скидання шляху до обробленого файлу
            image = ft.Image(src=selected_file_path, width=400, height=400)  # Створення об'єкта зображення
            left_image_container.content = image  # Відображення зображення в лівому контейнері
            right_image_container.content = None  # Очищення правого контейнера
            update_buttons(uploaded=True)  # Оновлення кнопок після завантаження файлу
            page.update()  # Оновлення сторінки

    # Функція для обробки зображення за допомогою моделі YOLOv5
    def process_image(e):
        global selected_file_path, processed_image_path, results
        if selected_file_path:
            results = model(selected_file_path)  # Отримання результатів розпізнавання
            results_img = results.render()[0]  # Отримання зображення з нанесеними об'єктами
            img_pil = Image.fromarray(results_img)  # Перетворення зображення в формат PIL

            # Збереження обробленого зображення у тимчасовий файл
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                img_pil.save(temp_img, format="JPEG")
                processed_image_path = temp_img.name  # Збереження шляху до тимчасового файлу

            right_image_container.content = ft.Image(src=processed_image_path, width=400, height=400)  # Відображення обробленого зображення
            update_buttons(processed=True)  # Оновлення кнопок після обробки зображення
            page.update()  # Оновлення сторінки

    # Функція для відображення деталей розпізнаних знаків
    def show_details(e):
        if results:
            recognized_classes = results.names  # Отримання імен розпізнаних класів
            recognized_labels = [recognized_classes[int(c)] for c in results.pred[0][:, -1]]  # Отримання міток розпізнаних класів
            unique_labels = list(set(recognized_labels))  # Визначення унікальних міток

            # Створення списку класів для відображення
            class_list = ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text(f"{label}:\n{class_mappings.get(label, 'Unknown')}", size=20),
                            ft.IconButton(icon=ft.icons.INFO, on_click=lambda e, label=label: show_sign_info(label))
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        spacing=10
                    ) for label in unique_labels
                ],
                spacing=10
            )

            # Додавання нового виду для відображення деталей
            page.views.append(
                ft.View(
                    route='/details',
                    controls=[
                        ft.AppBar(title=ft.Text('Details'), bgcolor='blue'),  # Додавання заголовка
                        ft.Row(
                            controls=[
                                ft.Image(src=processed_image_path, width=400, height=400),  # Відображення обробленого зображення
                                ft.Column(
                                    controls=[
                                        ft.Text('Recognized Classes', size=20, weight=ft.FontWeight.BOLD),  # Заголовок списку класів
                                        class_list  # Відображення списку класів
                                    ],
                                    alignment=ft.MainAxisAlignment.START,
                                    spacing=20
                                )
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=200
                        )
                    ],
                    vertical_alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=26
                )
            )
            page.update()  # Оновлення сторінки

    # Функція для відображення інформації про конкретний знак
    def show_sign_info(label):
        icon_path = f'icons/{label}.png'  # Формування шляху до іконки знаку
        if os.path.exists(icon_path):
            icon_image = ft.Image(src=icon_path, width=100, height=100)  # Відображення іконки, якщо вона існує
        else:
            icon_image = ft.Text('', size=20)  # Відображення порожнього тексту, якщо іконки немає

        meaning = meaning_mappings.get(label, 'Інформація про цей знак відсутня в базі даних.')  # Отримання значення знаку з мапінгу

        # Додавання нового виду для відображення інформації про знак
        page.views.append(
            ft.View(
                route=f'/details/{label}',
                controls=[
                    ft.AppBar(title=ft.Text('Information'), bgcolor='blue'),  # Додавання заголовка
                    ft.Column(
                        controls=[
                            icon_image,  # Відображення іконки
                            ft.Text(f"{label}", size=20),  # Відображення назви знаку
                            ft.Text(f"{meaning}", size=20)  # Відображення значення знаку
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=20
                    )
                ],
                vertical_alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=26
            )
        )
        page.update()  # Оновлення сторінки

    # Функція для оновлення кнопок в залежності від стану (завантажено/оброблено)
    def update_buttons(uploaded=False, processed=False):
        if uploaded:
            button_container.content = ft.Row(
                controls=[
                    ft.ElevatedButton(text='Upload another photo', on_click=upload_photo),  # Кнопка для завантаження іншого фото
                    ft.ElevatedButton(text='Process the image', on_click=process_image)  # Кнопка для обробки зображення
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20
            )
        elif processed:
            button_container.content = ft.Row(
                controls=[
                    ft.ElevatedButton(text='Upload another photo', on_click=upload_photo),  # Кнопка для завантаження іншого фото
                    ft.ElevatedButton(text='Details', on_click=show_details)  # Кнопка для відображення деталей
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20
            )
        page.update()  # Оновлення сторінки

    # Обробка зміни маршруту сторінки
    def route_change(e):
        page.views.clear()  # Очищення всіх виглядів сторінки

        # Додавання головного вигляду
        page.views.append(
            ft.View(
                route='/',
                controls=[
                    ft.AppBar(title=ft.Text('Home'), bgcolor='blue'),  # Додавання заголовка
                    ft.Text('Welcome to', width=550, size=22),  # Текст привітання
                    ft.Text('Traffic Sign Recognition App', height=125, size=30),  # Назва додатку
                    ft.ElevatedButton(text='Start', width=250, height=75, on_click=lambda _: page.go('/recognition'))  # Кнопка для початку розпізнавання
                ],
                vertical_alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=26
            )
        )

        # Додавання вигляду для розпізнавання
        if page.route == '/recognition':
            button_container.content = ft.ElevatedButton(text='Upload the photo', on_click=upload_photo)  # Початкова кнопка для завантаження фото

            page.views.append(
                ft.View(
                    route='/recognition',
                    controls=[
                        ft.AppBar(title=ft.Text('Recognition'), bgcolor='blue'),  # Додавання заголовка
                        ft.Row(
                            controls=[
                                left_image_container,  # Контейнер для завантаженого зображення
                                right_image_container  # Контейнер для обробленого зображення
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=200
                        ),
                        button_container  # Контейнер для кнопок
                    ],
                    vertical_alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=26
                )
            )

        # Додавання вигляду для деталей розпізнаних знаків
        if page.route.startswith('/details'):
            parts = page.route.split('/')
            if len(parts) == 3:
                label = parts[2]
                show_sign_info(label)  # Відображення інформації про конкретний знак
            else:
                show_details(None)  # Відображення загальних деталей

        page.update()  # Оновлення сторінки

    # Обробка повернення до попереднього вигляду
    def view_pop(e):
        page.views.pop()  # Видалення останнього вигляду
        top_view: ft.View = page.views[-1]  # Отримання верхнього вигляду
        page.go(top_view.route)  # Перехід до верхнього вигляду

    page.on_route_change = route_change  # Призначення обробника для зміни маршруту
    page.on_view_pop = view_pop  # Призначення обробника для повернення до попереднього вигляду
    page.go(page.route)  # Перехід до поточного маршруту

if __name__ == '__main__':
    ft.app(target=main)  # Запуск додатку