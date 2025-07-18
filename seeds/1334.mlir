module {
  func.func @main(%arg0: tensor<97x85x100x28x50xf32>, %arg1: tensor<5x2xi32>, %arg2: tensor<11xi1>) -> (tensor<97x170x100x28x50xf32>, tensor<11xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<10xindex>} : () -> !tosa.shape<10>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<97x85x100x28x50xf32>, !tosa.shape<10>, tensor<1xf32>) -> tensor<97x85x100x28x50xf32>
    %1 = tosa.floor %0 : (tensor<97x85x100x28x50xf32>) -> tensor<97x85x100x28x50xf32>
    %2 = tosa.concat %1, %1 {axis = 1 : i32} : (tensor<97x85x100x28x50xf32>, tensor<97x85x100x28x50xf32>) -> tensor<97x170x100x28x50xf32>
    %3 = tosa.logical_not %arg2 : (tensor<11xi1>) -> tensor<11xi1>
    return %2, %3 : tensor<97x170x100x28x50xf32>, tensor<11xi1>
  }
}
