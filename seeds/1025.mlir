module {
  func.func @main(%arg0: tensor<56x77x31x16x27x46xi8>, %arg1: tensor<15x44x21x48xi1>) -> (tensor<56x77x31x16x27x46xi1>, tensor<1x44x21x48xi1>, tensor<1x44x21x48xi1>) {
    %0 = tosa.identity %arg0 : (tensor<56x77x31x16x27x46xi8>) -> tensor<56x77x31x16x27x46xi8>
    %1 = tosa.maximum %0, %0 : (tensor<56x77x31x16x27x46xi8>, tensor<56x77x31x16x27x46xi8>) -> tensor<56x77x31x16x27x46xi8>
    %2 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<15x44x21x48xi1>) -> tensor<1x44x21x48xi1>
    %3 = tosa.greater %1, %0 : (tensor<56x77x31x16x27x46xi8>, tensor<56x77x31x16x27x46xi8>) -> tensor<56x77x31x16x27x46xi1>
    %4 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<1x44x21x48xi1>) -> tensor<1x44x21x48xi1>
    %5 = tosa.logical_not %2 : (tensor<1x44x21x48xi1>) -> tensor<1x44x21x48xi1>
    return %3, %4, %5 : tensor<56x77x31x16x27x46xi1>, tensor<1x44x21x48xi1>, tensor<1x44x21x48xi1>
  }
}
