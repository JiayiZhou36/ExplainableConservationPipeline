def evaluate_model(model, test_images):
    results = model.eval(test_images)
    return results
