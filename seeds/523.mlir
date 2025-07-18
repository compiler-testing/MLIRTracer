module {
  func.func @main(%arg0: tensor<79x31x19xi32>, %arg1: tensor<79x31x19xi32>, %arg2: tensor<56x35x44x39xf32>, %arg3: tensor<56x1x44x39xf32>, %arg4: tensor<90x84x76x84x87xi1>) -> (tensor<158x62x38xi32>, tensor<56x35x44x39xf32>, tensor<2x7644x220xf32>, tensor<90x84x76x84x87xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<79x31x19xi32>, tensor<79x31x19xi32>) -> tensor<79x31x19xi32>
    %1 = tosa.add %0, %0 : (tensor<79x31x19xi32>, tensor<79x31x19xi32>) -> tensor<79x31x19xi32>
    %2 = tosa.bitwise_or %1, %0 : (tensor<79x31x19xi32>, tensor<79x31x19xi32>) -> tensor<79x31x19xi32>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<79x31x19xi32>) -> tensor<79x31x19xi32>
    %t_4 = tosa.const_shape {values = dense<[ 2, 2, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.tile %3, %t_4 : (tensor<79x31x19xi32>, !tosa.shape<3>) -> tensor<158x62x38xi32>
    %5 = tosa.pow %arg2, %arg3 : (tensor<56x35x44x39xf32>, tensor<56x1x44x39xf32>) -> tensor<56x35x44x39xf32>
    %6 = tosa.bitwise_not %4 : (tensor<158x62x38xi32>) -> tensor<158x62x38xi32>
    %7 = tosa.logical_not %arg4 : (tensor<90x84x76x84x87xi1>) -> tensor<90x84x76x84x87xi1>
    %8 = tosa.exp %5 : (tensor<56x35x44x39xf32>) -> tensor<56x35x44x39xf32>
    %r_9 = tosa.const_shape {values = dense<[ 2, 7644, 220 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.reshape %5, %r_9 : (tensor<56x35x44x39xf32>, !tosa.shape<3>) -> tensor<2x7644x220xf32>
    %10 = tosa.logical_left_shift %7, %7 : (tensor<90x84x76x84x87xi1>, tensor<90x84x76x84x87xi1>) -> tensor<90x84x76x84x87xi1>
    return %6, %8, %9, %10 : tensor<158x62x38xi32>, tensor<56x35x44x39xf32>, tensor<2x7644x220xf32>, tensor<90x84x76x84x87xi1>
  }
}
