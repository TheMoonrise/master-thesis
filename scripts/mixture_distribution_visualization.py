import tkinter as tk
from tkinter import Tk, Scale, Label


def update_result(event=None):
    w1 = weight_01.get()
    w2 = weight_02.get()
    m1 = mean_01.get()
    m2 = mean_02.get()
    v1 = variance_01.get()
    v2 = variance_02.get()

    m = w1 * m1 + w2 * m2
    v = w1 * v1 + w1 * m1**2 + w2 * v2 + w2 * m2**2 - m**2

    result.config(text=f'Mean: {m:.2f}, Variance: {v:.2f}')


def update_weight_01(event):
    weight_02.set(1 - weight_01.get())
    update_result()


def update_weight_02(event):
    weight_01.set(1 - weight_02.get())
    update_result()


root = Tk()
root.title('Mixture Distribution Moments')

label_01 = Label(root, text='Primary Distribution')
label_01.pack(pady=(30, 0))

mean_01 = Scale(root, from_=0, to=10, length=500, tickinterval=1, resolution=0.1,
                orient=tk.HORIZONTAL, label='Mean', command=update_result)
mean_01.pack()

variance_01 = Scale(root, from_=1, to=10, length=500, tickinterval=1, resolution=0.1,
                    orient=tk.HORIZONTAL, label='Variance', command=update_result)
variance_01.pack()

weight_01 = Scale(root, from_=0, to=1, resolution=0.01, length=500, tickinterval=1,
                  orient=tk.HORIZONTAL, label='Weight', command=update_weight_01)
weight_01.pack()

label_02 = Label(root, text='Secondary Distribution')
label_02.pack(pady=(30, 0))

mean_02 = Scale(root, from_=0, to=10, length=500, tickinterval=1, resolution=0.1,
                orient=tk.HORIZONTAL, label='Mean', command=update_result)
mean_02.pack()

variance_02 = Scale(root, from_=1, to=10, length=500, tickinterval=1, resolution=0.1,
                    orient=tk.HORIZONTAL, label='Variance', command=update_result)
variance_02.pack()

weight_02 = Scale(root, from_=0, to=1, resolution=0.01, length=500, tickinterval=1,
                  orient=tk.HORIZONTAL, label='Weight', command=update_weight_02)
weight_02.pack()

result = Label(root, text='Hello World!')
result.pack(pady=(30, 30))

weight_01.set(1)

update_result()
root.mainloop()
