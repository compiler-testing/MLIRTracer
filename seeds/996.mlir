module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<34xi16>, %arg3: tensor<87xi16>, %arg4: tensor<92x99x1x100x90x46xf32>, %arg5: tensor<92x99x1x100x1x46xf32>, %arg6: tensor<i32>, %arg7: tensor<i32>) -> (tensor<i32>, tensor<184x99x1x100x90x46xf32>, tensor<i1>, tensor<121xi16>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.concat %arg2, %arg3 {axis = 0 : i32} : (tensor<34xi16>, tensor<87xi16>) -> tensor<121xi16>
    %2 = tosa.minimum %arg4, %arg5 : (tensor<92x99x1x100x90x46xf32>, tensor<92x99x1x100x1x46xf32>) -> tensor<92x99x1x100x90x46xf32>
    %3 = tosa.intdiv %arg6, %arg7 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = tosa.log %2 : (tensor<92x99x1x100x90x46xf32>) -> tensor<92x99x1x100x90x46xf32>
    %5 = tosa.concat %4, %4 {axis = 0 : i32} : (tensor<92x99x1x100x90x46xf32>, tensor<92x99x1x100x90x46xf32>) -> tensor<184x99x1x100x90x46xf32>
    %6 = tosa.logical_and %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.clz %1 : (tensor<121xi16>) -> tensor<121xi16>
    return %3, %5, %6, %7 : tensor<i32>, tensor<184x99x1x100x90x46xf32>, tensor<i1>, tensor<121xi16>
  }
}
