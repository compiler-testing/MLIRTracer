module {
  func.func @main(%arg0: tensor<95x68x84x32x6x77xf32>, %arg1: tensor<78xi8>) -> (tensor<78xi1>, tensor<1xi1>, tensor<95x68x84x32x6x77xf32>) {
    %0 = tosa.clamp %arg0 {min_val = -1.900000e+01 : f32, max_val = 9.000000e+00 : f32} : (tensor<95x68x84x32x6x77xf32>) -> tensor<95x68x84x32x6x77xf32>
    %1 = tosa.bitwise_not %arg1 : (tensor<78xi8>) -> tensor<78xi8>
    %2 = tosa.greater_equal %1, %1 : (tensor<78xi8>, tensor<78xi8>) -> tensor<78xi1>
    %3 = tosa.ceil %0 : (tensor<95x68x84x32x6x77xf32>) -> tensor<95x68x84x32x6x77xf32>
    %4 = tosa.logical_not %2 : (tensor<78xi1>) -> tensor<78xi1>
    %5 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<78xi1>) -> tensor<1xi1>
    %6 = tosa.sigmoid %3 : (tensor<95x68x84x32x6x77xf32>) -> tensor<95x68x84x32x6x77xf32>
    return %4, %5, %6 : tensor<78xi1>, tensor<1xi1>, tensor<95x68x84x32x6x77xf32>
  }
}
