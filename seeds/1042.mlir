module {
  func.func @main(%arg0: tensor<37x67x82xi16>, %arg1: tensor<37x67x1xi16>, %arg2: tensor<94x84x73x37x99x53xf32>) -> (tensor<37x67x82xi16>, tensor<94x84x73x37x99x53xf32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<37x67x82xi16>, tensor<37x67x1xi16>) -> tensor<37x67x82xi16>
    %1 = tosa.identity %0 : (tensor<37x67x82xi16>) -> tensor<37x67x82xi16>
    %2 = tosa.tanh %arg2 : (tensor<94x84x73x37x99x53xf32>) -> tensor<94x84x73x37x99x53xf32>
    return %1, %2 : tensor<37x67x82xi16>, tensor<94x84x73x37x99x53xf32>
  }
}
