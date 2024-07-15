# Example Pytorch Model to ONNXRuntime-Web

Proyek ini bertujuan untuk melatih model deep learning menggunakan dataset MNIST untuk mengenali angka tulisan tangan. Model akan diekspor ke format ONNX untuk integrasi web sederhana yang kita gambar sendiri di canvas dan menampilkan hasil prediksi nya dengan model yang telah dibuat


## Build Model

Clone project

```bash
  git clone https://github.com/naufalahnaf17/example-torch-onnx.git
```

Masuk ke direktori model dan aktifkan Virtual ENV

```bash
    cd model
    python -m venv env
    source env/bin/activate
```

Install requirements.txt dan Training model

```bash
    pip install -r requirements.txt
    python main.py
```

### Apa saja yang bisa diganti pada proses training ?
- `num_epochs = 10 (default)`
- `criterion = CrossEntropyLoss`
- `optimizer = Adam`
- `learning rate = 1e-3 / 0.003`

### Output Folder
- /data `(MNIST Dataset)`
- /env `Virtual Env`
- /output `(model.pth dan model.onnx)`

### Akurasi Training
98% Akurasi

![Akurasi Model](screenshot/ss_1.png)

## Build ONNXRuntime-Web 

Soon



## Authors

- [@naufalahnaf17](https://github.com/naufalahnaf17)

## License

[MIT](https://choosealicense.com/licenses/mit/)


