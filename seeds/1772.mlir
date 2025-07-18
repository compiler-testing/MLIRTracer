module {
  func.func @main(%arg0: tensor<40x23x95x2x12x50xf32>, %arg1: tensor<61x41x93x47x73x11xi32>, %arg2: tensor<1x1x1x1x1x11xi32>) -> (tensor<40x23x95x2x12x50xf32>, tensor<61x41x93x47x73x11xi32>) {
    %0 = tosa.ceil %arg0 : (tensor<40x23x95x2x12x50xf32>) -> tensor<40x23x95x2x12x50xf32>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<61x41x93x47x73x11xi32>, tensor<1x1x1x1x1x11xi32>) -> tensor<61x41x93x47x73x11xi32>
    return %0, %1 : tensor<40x23x95x2x12x50xf32>, tensor<61x41x93x47x73x11xi32>
  }
}
