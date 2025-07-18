module {
  func.func @main(%arg0: tensor<78x76x78x12xi8>, %arg1: tensor<75x44x97x31x18x15xf32>, %arg2: tensor<39xi1>, %arg3: tensor<39xi1>) -> (tensor<39xi1>, tensor<75x44x97x31x18x15xf32>, tensor<78x76x78x12xi1>) {
    %0 = tosa.clamp %arg0 {min_val = -39 : i8, max_val = -18 : i8} : (tensor<78x76x78x12xi8>) -> tensor<78x76x78x12xi8>
    %1 = tosa.reciprocal %arg1 : (tensor<75x44x97x31x18x15xf32>) -> tensor<75x44x97x31x18x15xf32>
    %2 = tosa.reverse %0 {axis = 3 : i32} : (tensor<78x76x78x12xi8>) -> tensor<78x76x78x12xi8>
    %3 = tosa.logical_and %arg2, %arg3 : (tensor<39xi1>, tensor<39xi1>) -> tensor<39xi1>
    %4 = tosa.clamp %1 {min_val = -1.800000e+01 : f32, max_val = -3.000000e+00 : f32} : (tensor<75x44x97x31x18x15xf32>) -> tensor<75x44x97x31x18x15xf32>
    %5 = tosa.equal %2, %2 : (tensor<78x76x78x12xi8>, tensor<78x76x78x12xi8>) -> tensor<78x76x78x12xi1>
    %6 = tosa.reverse %5 {axis = 1 : i32} : (tensor<78x76x78x12xi1>) -> tensor<78x76x78x12xi1>
    return %3, %4, %6 : tensor<39xi1>, tensor<75x44x97x31x18x15xf32>, tensor<78x76x78x12xi1>
  }
}
