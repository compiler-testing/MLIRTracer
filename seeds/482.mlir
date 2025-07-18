module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<3xi8>, %arg2: tensor<49xi8>, %arg3: tensor<31x57xf32>) -> (tensor<i16>, tensor<52xi1>, tensor<31x57xf32>, tensor<52xi1>, tensor<52xi1>) {
    %0 = tosa.clz %arg0 : (tensor<i16>) -> tensor<i16>
    %1 = tosa.concat %arg1, %arg2 {axis = 0 : i32} : (tensor<3xi8>, tensor<49xi8>) -> tensor<52xi8>
    %2 = tosa.greater %1, %1 : (tensor<52xi8>, tensor<52xi8>) -> tensor<52xi1>
    %3 = tosa.ceil %arg3 : (tensor<31x57xf32>) -> tensor<31x57xf32>
    %4 = tosa.reciprocal %3 : (tensor<31x57xf32>) -> tensor<31x57xf32>
    %5 = tosa.reverse %4 {axis = 1 : i32} : (tensor<31x57xf32>) -> tensor<31x57xf32>
    %6 = tosa.greater %1, %1 : (tensor<52xi8>, tensor<52xi8>) -> tensor<52xi1>
    %7 = tosa.greater %1, %1 : (tensor<52xi8>, tensor<52xi8>) -> tensor<52xi1>
    return %0, %2, %5, %6, %7 : tensor<i16>, tensor<52xi1>, tensor<31x57xf32>, tensor<52xi1>, tensor<52xi1>
  }
}
