module {
  func.func @main(%arg0: tensor<86x21x85x19xi1>, %arg1: tensor<61xi8>, %arg2: tensor<61xi8>, %arg3: tensor<f32>) -> (tensor<86x21x85x19xi1>, tensor<61xi1>, tensor<f32>, tensor<1xi1>) {
    %0 = tosa.logical_not %arg0 : (tensor<86x21x85x19xi1>) -> tensor<86x21x85x19xi1>
    %1 = tosa.minimum %arg1, %arg2 : (tensor<61xi8>, tensor<61xi8>) -> tensor<61xi8>
    %2 = tosa.abs %1 : (tensor<61xi8>) -> tensor<61xi8>
    %3 = tosa.greater %2, %1 : (tensor<61xi8>, tensor<61xi8>) -> tensor<61xi1>
    %4 = tosa.floor %arg3 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.greater_equal %2, %1 : (tensor<61xi8>, tensor<61xi8>) -> tensor<61xi1>
    %6 = tosa.reduce_any %5 {axis = 0 : i32} : (tensor<61xi1>) -> tensor<1xi1>
    %7 = tosa.pow %4, %4 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = tosa.logical_and %6, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %0, %3, %7, %8 : tensor<86x21x85x19xi1>, tensor<61xi1>, tensor<f32>, tensor<1xi1>
  }
}
