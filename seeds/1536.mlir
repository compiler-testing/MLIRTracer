module {
  func.func @main(%arg0: tensor<20xf32>, %arg1: tensor<78x69x75x93x18x5xi16>, %arg2: tensor<78x1x1x1x18x5xi16>) -> (tensor<20xf32>, tensor<78x69x75x93x18x5xi16>) {
    %0 = tosa.ceil %arg0 : (tensor<20xf32>) -> tensor<20xf32>
    %1 = tosa.logical_right_shift %arg1, %arg2 : (tensor<78x69x75x93x18x5xi16>, tensor<78x1x1x1x18x5xi16>) -> tensor<78x69x75x93x18x5xi16>
    return %0, %1 : tensor<20xf32>, tensor<78x69x75x93x18x5xi16>
  }
}
