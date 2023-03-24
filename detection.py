import turicreate as tc
#load the model
model = tc.load_model('HongKong-dollar.model')

# load the image
image_data = tc.image_analysis.load_images('./50dollar.jpg')

prediction1 = model.predict(image_data, output_type='class')

# print the result
print(prediction1[0])