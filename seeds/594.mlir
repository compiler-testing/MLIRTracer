module {
  func.func @main(%arg0: tensor<7x5xf32>) -> tensor<21x20xi1> {
    %0 = tosa.log %arg0 : (tensor<7x5xf32>) -> tensor<7x5xf32>
    %1 = tosa.equal %0, %0 : (tensor<7x5xf32>, tensor<7x5xf32>) -> tensor<7x5xi1>
    %t_2 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.tile %1, %t_2 : (tensor<7x5xi1>, !tosa.shape<2>) -> tensor<21x10xi1>
    %3 = tosa.concat %2, %2 {axis = 1 : i32} : (tensor<21x10xi1>, tensor<21x10xi1>) -> tensor<21x20xi1>
    %4 = tosa.clz %3 : (tensor<21x20xi1>) -> tensor<21x20xi1>
    %5 = tosa.logical_and %4, %4 : (tensor<21x20xi1>, tensor<21x20xi1>) -> tensor<21x20xi1>
    %6 = tosa.logical_not %5 : (tensor<21x20xi1>) -> tensor<21x20xi1>
    return %6 : tensor<21x20xi1>
  }
}
