module {
  func.func @main(%arg0: tensor<54x27x86x27x24x37xf32>, %arg1: tensor<21x3x84x63xi64>, %arg2: tensor<17x30xi1>, %arg3: tensor<17x30xi1>, %arg4: tensor<29x72x35x35xi32>, %arg5: tensor<29x1x1x35xi32>) -> (tensor<54x27x86x27x24x37xf32>, tensor<21x3x252x126xi1>, tensor<29x72x35x35xi32>, tensor<1x90xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<54x27x86x27x24x37xf32>) -> tensor<54x27x86x27x24x37xf32>
    %t_1 = tosa.const_shape {values = dense<[ 1, 1, 3, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.tile %arg1, %t_1 : (tensor<21x3x84x63xi64>, !tosa.shape<4>) -> tensor<21x3x252x126xi64>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<17x30xi1>, tensor<17x30xi1>) -> tensor<17x30xi1>
    %3 = tosa.equal %1, %1 : (tensor<21x3x252x126xi64>, tensor<21x3x252x126xi64>) -> tensor<21x3x252x126xi1>
    %4 = tosa.sub %2, %2 : (tensor<17x30xi1>, tensor<17x30xi1>) -> tensor<17x30xi1>
    %5 = tosa.intdiv %arg4, %arg5 : (tensor<29x72x35x35xi32>, tensor<29x1x1x35xi32>) -> tensor<29x72x35x35xi32>
    %t_6 = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.tile %4, %t_6 : (tensor<17x30xi1>, !tosa.shape<2>) -> tensor<34x90xi1>
    %7 = tosa.reduce_any %6 {axis = 0 : i32} : (tensor<34x90xi1>) -> tensor<1x90xi1>
    return %0, %3, %5, %7 : tensor<54x27x86x27x24x37xf32>, tensor<21x3x252x126xi1>, tensor<29x72x35x35xi32>, tensor<1x90xi1>
  }
}
