
from utils.utils import plot_accuracy_loss

def question_1(model, train, valid, test, person, batch_size=64, epochs=50, everyone_epochs=40):
  X_train, y_train, person_train = train
  X_valid, y_valid, person_valid = valid
  X_test,  y_test,  person_test  = test

  # Separate this individual's training samples
  train_mask = (person_train == person).flatten()
  X_train_me = X_train[train_mask]
  y_train_me = y_train[train_mask]

  # Forget about validating with anybody else
  valid_mask = (person_valid == person).flatten()
  X_valid = X_valid[valid_mask]
  y_valid = y_valid[valid_mask]

  # Forget about testing with anybody else
  test_mask = (person_test == person).flatten()
  X_test = X_test[test_mask]
  y_test = y_test[test_mask]

  # Train with entire training set
  if everyone_epochs > 0:
    everyone_results = model.fit(X_train, y_train,
        batch_size=batch_size,
        epochs=everyone_epochs,
        validation_data=(X_valid, y_valid), verbose=False)
    # plot_accuracy_loss(everyone_results.history)

  # Train with just this person
  if epochs - everyone_epochs > 0:
    me_results = model.fit(X_train_me, y_train_me,
        batch_size=batch_size,
        epochs=epochs-everyone_epochs,
        validation_data=(X_valid, y_valid), verbose=False)
    # plot_accuracy_loss(me_results.history)
    
  score = model.evaluate(X_test, y_test, verbose=False)
  print(f'Test accuracy of the model ({everyone_epochs}/{epochs}) on person-{person}:', round(score[1],2))