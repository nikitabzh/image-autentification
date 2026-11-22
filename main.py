"""Запуск: python main.py"""
import io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from core import Database, prepare
from neural_faces import available as sface_available


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Лабораторная 4 — Идентификация по изображению лица')
        self.geometry('1000x650')
        self.minsize(850,550)
        self.db = Database()
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.status = tk.StringVar(value='Зарегистрируйте пользователя или выберите фото для идентификации.')
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both',expand=True,padx=15,pady=15)
        identify = ttk.Frame(notebook,padding=16)
        people = ttk.Frame(notebook,padding=16)
        notebook.add(identify,text='Идентификация и вход')
        notebook.add(people,text='База пользователей')
        ttk.Label(identify,text='Идентификация пользователя по фотографии',font=('Segoe UI',16)).pack(anchor='w')
        ttk.Label(identify,text='Выберите фото с одним лицом. Для входа дополнительно потребуется пароль.').pack(anchor='w',pady=12)
        controls = ttk.Frame(identify)
        controls.pack(fill='x')
        ttk.Button(controls,text='Выбрать фото и распознать',command=self.identify).pack(side='left')
        ttk.Label(controls,text='  Порог (авто или число):').pack(side='left')
        self.threshold = tk.StringVar(value='авто')
        ttk.Entry(controls,textvariable=self.threshold,width=7).pack(side='left')
        ttk.Button(controls,text='Обучить модель',command=self.train_model).pack(side='left',padx=12)
        self.model_info = tk.StringVar()
        ttk.Label(identify,textvariable=self.model_info,wraplength=850).pack(anchor='w',pady=8)
        self.preview = ttk.Label(identify)
        self.preview.pack(pady=15)
        self.result = tk.StringVar(value='Фото ещё не выбрано.')
        ttk.Label(identify,textvariable=self.result,wraplength=800,font=('Segoe UI',12)).pack(anchor='w')
        ttk.Label(identify,text='Для проверки выберите фотографию, которой нет среди эталонов.',wraplength=800).pack(anchor='w',pady=15)
        bar = ttk.Frame(people)
        bar.pack(fill='x',pady=(0,10))
        for text, command in [('Регистрация',self.register),('Изменить',self.edit),('Фотографии',self.gallery),('Удалить',self.delete),('Обучить модель',self.train_model)]:
            ttk.Button(bar,text=text,command=command).pack(side='left',padx=3)
        self.tree = ttk.Treeview(people,columns=('login','name','count','created'),show='headings',selectmode='browse')
        for col,title,width in [('login','Логин',140),('name','ФИО',280),('count','Фото',60),('created','Создано (UTC)',170)]:
            self.tree.heading(col,text=title)
            self.tree.column(col,width=width)
        scroll = ttk.Scrollbar(people,orient='vertical',command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y')
        self.tree.pack(fill='both',expand=True)
        ttk.Label(self,textvariable=self.status,wraplength=950).pack(fill='x',padx=15,pady=(0,12))
        self.refresh()

    def close(self):
        self.db.conn.close()
        self.destroy()

    def train_model(self):
        try:
            model = self.db.train(use_sface=sface_available())
            self.threshold.set('авто')
            self.refresh()
            r = model['report']
            far = f"{r['far']:.1%}" if r['far'] is not None else 'не оценён: нужен второй человек'
            split = (f"Эталонов: {r['train']}\nПроверка с исключением снимка: {r['validation']}\n"
                     if model.get('backend') == 'sface'
                     else f"Обучающих фото: {r['train']}\nПроверочных фото: {r['validation']}\n")
            messagebox.showinfo('Обучение завершено',
                f"Пользователей: {r['users']}\n{split}Порог: {model['threshold']:.4f}\n"
                f"Допуск чужого по порогу: {far}\nОтказ своему по порогу: {r['frr']:.1%}\n"
                'Это результаты подбора порога, а не независимая оценка точности. '
                'Проверьте модель на новых снимках.')
            self.status.set('Модель обучена и сохранена в базе. При распознавании используется автоматически.')
        except Exception as exc:
            messagebox.showerror('Обучение не выполнено',str(exc))

    def refresh(self):
        self.model_info.set(self.db.model_state()[1])
        self.result.set('База обновлена. Выберите фото заново для идентификации.' if self.db.users() else 'В базе пока нет пользователей. Откройте вкладку «База пользователей» и зарегистрируйтесь.')
        self.preview.configure(image='')
        self.tree.delete(*self.tree.get_children())
        for u in self.db.users():
            self.tree.insert('', 'end',iid=str(u['id']),values=(u['login'],u['name'],u['count'],u['created']))

    def selected(self):
        ids = self.tree.selection()
        if not ids:
            messagebox.showinfo('База','Выберите пользователя.')
            return None
        return int(ids[0])

    def register(self, initial=()):
        self.editor(photos=list(initial))

    def edit(self):
        uid = self.selected()
        if uid is not None:
            self.editor(uid)

    def editor(self, uid=None, photos=None):
        photos = photos or []
        row = next((u for u in self.db.users() if u['id']==uid),None)
        win = tk.Toplevel(self)
        win.title('Редактирование' if row else 'Регистрация пользователя')
        win.geometry('520x330')
        win.transient(self)
        win.grab_set()
        fields = []
        for i,(label,value) in enumerate([('Логин',row['login'] if row else ''),('ФИО',row['name'] if row else ''),('Новый пароль (при изменении можно оставить пустым)','')]):
            ttk.Label(win,text=label).pack(anchor='w',padx=16,pady=(8,0))
            v = tk.StringVar(value=value)
            ttk.Entry(win,textvariable=v,show='*' if i==2 else '').pack(fill='x',padx=16)
            fields.append(v)
        info = tk.StringVar(value=f'Новых фотографий: {len(photos)}')
        ttk.Label(win,textvariable=info).pack(pady=8)
        def add():
            paths = filedialog.askopenfilenames(parent=win,filetypes=[('Фотографии','*.jpg *.jpeg *.png *.bmp'),('Все файлы','*.*')])
            # Обрабатываем весь пакет до изменения списка или базы.
            try:
                prepared = [self.enrollment_photo(prepare(p, use_sface=sface_available())) for p in paths]
                photos.extend(prepared)
                info.set(f'Новых фотографий: {len(photos)}')
            except Exception as exc:
                messagebox.showerror('Фото не добавлены',str(exc),parent=win)
        def save():
            try:
                self.db.save(*(v.get() for v in fields),photos,uid)
                self.refresh()
                self.status.set('Данные пользователя сохранены.')
                win.destroy()
            except Exception as exc:
                messagebox.showerror('Не удалось сохранить',str(exc),parent=win)
        ttk.Button(win,text='Добавить фотографии (можно выбрать несколько)',command=add).pack(pady=4)
        ttk.Button(win,text='Сохранить',command=save).pack(pady=8)

    def delete(self):
        uid = self.selected()
        if uid is not None and messagebox.askyesno('Удаление','Удалить выбранного пользователя и все его фотографии?'):
            self.db.delete(uid)
            self.refresh()
            self.status.set('Пользователь и его фотографии удалены.')

    def gallery(self):
        uid = self.selected()
        if uid is None:
            return
        win = tk.Toplevel(self)
        win.title('Фотографии пользователя')
        win.geometry('600x460')
        listing = tk.Listbox(win,exportselection=False)
        listing.pack(fill='x',padx=12,pady=12)
        preview = ttk.Label(win)
        preview.pack()
        rows = []
        def show(event=None):
            if listing.curselection():
                face = Image.open(io.BytesIO(rows[listing.curselection()[0]]['face']))
                face.thumbnail((260,260))
                preview.image = ImageTk.PhotoImage(face)
                preview.configure(image=preview.image)
        def reload():
            rows[:] = self.db.photos(uid)
            listing.delete(0,'end')
            for p in rows:
                listing.insert('end',f"{p['id']} — {p['filename']}")
            if rows:
                listing.selection_set(0)
                show()
        def remove():
            if listing.curselection() and messagebox.askyesno('Удаление','Удалить выбранное фото?',parent=win):
                try:
                    self.db.delete_photo(rows[listing.curselection()[0]]['id'],uid)
                    reload()
                    self.refresh()
                except ValueError as exc:
                    messagebox.showerror('Удаление',str(exc),parent=win)
        listing.bind('<<ListboxSelect>>',show)
        ttk.Button(win,text='Удалить фото',command=remove).pack(pady=8)
        reload()

    @staticmethod
    def enrollment_photo(photo):
        # Сохраняем LBP для старых моделей. SFace обучается по RGB оригиналу.
        from core import descriptor
        return (*photo[:3], descriptor(Image.open(io.BytesIO(photo[2]))).tobytes())

    def identify(self):
        path = filedialog.askopenfilename(filetypes=[('Фотографии','*.jpg *.jpeg *.png *.bmp'),('Все файлы','*.*')])
        if not path:
            return
        try:
            value = self.threshold.get().strip().lower()
            threshold = None if value in ('', 'авто', 'auto') else float(value.replace(',','.'))
            model, state = self.db.model_state()
            if model is None and self.db.has_model():
                raise ValueError(state)
            use_sface = model is not None and model.get('backend') == 'sface'
            photo = prepare(path, use_sface=use_sface)
            face = Image.open(io.BytesIO(photo[2]))
            face.thumbnail((270,270))
            self.preview.image = ImageTk.PhotoImage(face)
            self.preview.configure(image=self.preview.image)
            user, score, reason = self.db.identify(photo[3],threshold)
            self.result.set(reason + (f' Расстояние: {score:.4f} (меньше — лучше).' if score is not None else ''))
            if user is None:
                if messagebox.askyesno('Пользователь не определён',reason+'\nОткрыть регистрацию с этим фото?'):
                    self.register([self.enrollment_photo(photo) if use_sface else photo])
                return
            self.result.set(f"Распознан: {user['name']} ({user['login']}). Расстояние: {score:.4f}.")
            password = simpledialog.askstring('Аутентификация',f"Пользователь {user['login']}. Введите пароль:",show='*',parent=self)
            if password is None:
                self.status.set('Лицо распознано. Вход отменён.')
            elif self.db.check_password(user['id'],password):
                self.status.set(f"Аутентификация успешна: {user['name']}.")
                messagebox.showinfo('Вход','Лицо и пароль подтверждены. Вход выполнен.')
            else:
                self.status.set('Вход отклонён: неверный пароль.')
                messagebox.showerror('Вход','Неверный пароль.')
        except Exception as exc:
            messagebox.showerror('Ошибка распознавания',str(exc))


if __name__ == '__main__':
    App().mainloop()

