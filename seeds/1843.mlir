module {
  func.func @main(%arg0: tensor<97x39xf32>) -> tensor<1x39xf32> {
    %0 = tosa.floor %arg0 : (tensor<97x39xf32>) -> tensor<97x39xf32>
    %1 = tosa.floor %0 : (tensor<97x39xf32>) -> tensor<97x39xf32>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<97x39xf32>) -> tensor<1x39xf32>
    return %2 : tensor<1x39xf32>
  }
}
