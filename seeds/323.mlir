module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<37x78x94x98xi64>, %arg3: tensor<1x78x94x98xi64>, %arg4: tensor<95xf32>) -> (tensor<i1>, tensor<37x98x78x94xi1>, tensor<37x78x94x98xi1>, tensor<95xf32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.logical_xor %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.clz %1 : (tensor<i1>) -> tensor<i1>
    %3 = tosa.greater %arg2, %arg3 : (tensor<37x78x94x98xi64>, tensor<1x78x94x98xi64>) -> tensor<37x78x94x98xi1>
    %4 = tosa.reverse %3 {axis = 1 : i32} : (tensor<37x78x94x98xi1>) -> tensor<37x78x94x98xi1>
    %5 = tosa.sub %4, %4 : (tensor<37x78x94x98xi1>, tensor<37x78x94x98xi1>) -> tensor<37x78x94x98xi1>
    %6 = tosa.logical_not %3 : (tensor<37x78x94x98xi1>) -> tensor<37x78x94x98xi1>
    %7 = tosa.logical_xor %6, %6 : (tensor<37x78x94x98xi1>, tensor<37x78x94x98xi1>) -> tensor<37x78x94x98xi1>
    %8 = tosa.reverse %5 {axis = 2 : i32} : (tensor<37x78x94x98xi1>) -> tensor<37x78x94x98xi1>
    %9 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %10 = tosa.transpose %8 {perms = array<i32: 0, 3, 1, 2>} : (tensor<37x78x94x98xi1>) -> tensor<37x98x78x94xi1>
    %11 = tosa.add %10, %10 : (tensor<37x98x78x94xi1>, tensor<37x98x78x94xi1>) -> tensor<37x98x78x94xi1>
    %12 = tosa.reverse %7 {axis = 3 : i32} : (tensor<37x78x94x98xi1>) -> tensor<37x78x94x98xi1>
    %13 = tosa.floor %arg4 : (tensor<95xf32>) -> tensor<95xf32>
    return %2, %11, %12, %13 : tensor<i1>, tensor<37x98x78x94xi1>, tensor<37x78x94x98xi1>, tensor<95xf32>
  }
}
