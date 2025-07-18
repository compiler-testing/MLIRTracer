module {
  func.func @main(%arg0: tensor<86x97x66x67x81x82xi64>, %arg1: tensor<52x2x79x47x42xi1>, %arg2: tensor<1x2x79x47x1xi1>, %arg3: tensor<37x65x12xi64>, %arg4: tensor<2x91x97xf32>) -> (tensor<86x97x66x67x81x82xi64>, tensor<52x2x79x47x42xi1>, tensor<2x91x97xf32>, tensor<37x65x1xi64>) {
    %0 = tosa.clamp %arg0 {min_val = -22 : i64, max_val = -13 : i64} : (tensor<86x97x66x67x81x82xi64>) -> tensor<86x97x66x67x81x82xi64>
    %1 = tosa.maximum %0, %0 : (tensor<86x97x66x67x81x82xi64>, tensor<86x97x66x67x81x82xi64>) -> tensor<86x97x66x67x81x82xi64>
    %2 = tosa.logical_xor %arg1, %arg2 : (tensor<52x2x79x47x42xi1>, tensor<1x2x79x47x1xi1>) -> tensor<52x2x79x47x42xi1>
    %3 = tosa.reduce_min %arg3 {axis = 2 : i32} : (tensor<37x65x12xi64>) -> tensor<37x65x1xi64>
    %4 = tosa.reverse %3 {axis = 1 : i32} : (tensor<37x65x1xi64>) -> tensor<37x65x1xi64>
    %5 = tosa.sigmoid %arg4 : (tensor<2x91x97xf32>) -> tensor<2x91x97xf32>
    %6 = tosa.minimum %4, %4 : (tensor<37x65x1xi64>, tensor<37x65x1xi64>) -> tensor<37x65x1xi64>
    return %1, %2, %5, %6 : tensor<86x97x66x67x81x82xi64>, tensor<52x2x79x47x42xi1>, tensor<2x91x97xf32>, tensor<37x65x1xi64>
  }
}
