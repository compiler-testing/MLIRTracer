module {
  func.func @main(%arg0: tensor<5x27x47x31xi16>, %arg1: tensor<80xf32>, %arg2: tensor<i1>, %arg3: tensor<i1>) -> (tensor<5x27x1x31xi16>, tensor<80xf32>, tensor<5x27x47x31xi16>, tensor<i1>) {
    %0 = tosa.abs %arg0 : (tensor<5x27x47x31xi16>) -> tensor<5x27x47x31xi16>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<5x27x47x31xi16>, tensor<5x27x47x31xi16>) -> tensor<5x27x47x31xi16>
    %2 = tosa.ceil %arg1 : (tensor<80xf32>) -> tensor<80xf32>
    %3 = tosa.reduce_min %1 {axis = 2 : i32} : (tensor<5x27x47x31xi16>) -> tensor<5x27x1x31xi16>
    %4 = tosa.rsqrt %2 : (tensor<80xf32>) -> tensor<80xf32>
    %5 = tosa.bitwise_not %1 : (tensor<5x27x47x31xi16>) -> tensor<5x27x47x31xi16>
    %6 = tosa.logical_and %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %3, %4, %5, %6 : tensor<5x27x1x31xi16>, tensor<80xf32>, tensor<5x27x47x31xi16>, tensor<i1>
  }
}
