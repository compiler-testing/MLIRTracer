module {
  func.func @main(%arg0: tensor<37x47x30x3xi8>, %arg1: tensor<37x47x30x3xi8>, %arg2: tensor<49x93xf32>, %arg3: tensor<29xi1>, %arg4: tensor<29xi1>) -> (tensor<49x93xf32>, tensor<49x93xf32>, tensor<37x47x30x3xi8>, tensor<29xi1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<37x47x30x3xi8>, tensor<37x47x30x3xi8>) -> tensor<37x47x30x3xi8>
    %1 = tosa.exp %arg2 : (tensor<49x93xf32>) -> tensor<49x93xf32>
    %2 = tosa.bitwise_and %0, %0 : (tensor<37x47x30x3xi8>, tensor<37x47x30x3xi8>) -> tensor<37x47x30x3xi8>
    %3 = tosa.log %1 : (tensor<49x93xf32>) -> tensor<49x93xf32>
    %4 = tosa.log %1 : (tensor<49x93xf32>) -> tensor<49x93xf32>
    %5 = tosa.bitwise_not %2 : (tensor<37x47x30x3xi8>) -> tensor<37x47x30x3xi8>
    %6 = tosa.logical_xor %arg3, %arg4 : (tensor<29xi1>, tensor<29xi1>) -> tensor<29xi1>
    %7 = tosa.add %6, %6 : (tensor<29xi1>, tensor<29xi1>) -> tensor<29xi1>
    return %3, %4, %5, %7 : tensor<49x93xf32>, tensor<49x93xf32>, tensor<37x47x30x3xi8>, tensor<29xi1>
  }
}
