module {
  func.func @main(%arg0: tensor<99x70x29x64x92x15xi64>, %arg1: tensor<1x1x1x64x1x15xi64>, %arg2: tensor<19x38x39x52x46xi32>, %arg3: tensor<1x1x1x1x1xi32>, %arg4: tensor<51x40x50x78x55x74xf32>) -> (tensor<19x38x39x52x46xi32>, tensor<99x70x29x64x92x15xi1>, tensor<51x40x50x78x55x74xi1>, tensor<51x40x50x78x55x74xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<99x70x29x64x92x15xi64>, tensor<1x1x1x64x1x15xi64>) -> tensor<99x70x29x64x92x15xi1>
    %1 = tosa.logical_not %0 : (tensor<99x70x29x64x92x15xi1>) -> tensor<99x70x29x64x92x15xi1>
    %2 = tosa.intdiv %arg2, %arg3 : (tensor<19x38x39x52x46xi32>, tensor<1x1x1x1x1xi32>) -> tensor<19x38x39x52x46xi32>
    %3 = tosa.sub %2, %2 : (tensor<19x38x39x52x46xi32>, tensor<19x38x39x52x46xi32>) -> tensor<19x38x39x52x46xi32>
    %4 = tosa.rsqrt %arg4 : (tensor<51x40x50x78x55x74xf32>) -> tensor<51x40x50x78x55x74xf32>
    %5 = tosa.logical_xor %1, %0 : (tensor<99x70x29x64x92x15xi1>, tensor<99x70x29x64x92x15xi1>) -> tensor<99x70x29x64x92x15xi1>
    %6 = tosa.greater %4, %4 : (tensor<51x40x50x78x55x74xf32>, tensor<51x40x50x78x55x74xf32>) -> tensor<51x40x50x78x55x74xi1>
    %7 = tosa.pow %4, %4 : (tensor<51x40x50x78x55x74xf32>, tensor<51x40x50x78x55x74xf32>) -> tensor<51x40x50x78x55x74xf32>
    return %3, %5, %6, %7 : tensor<19x38x39x52x46xi32>, tensor<99x70x29x64x92x15xi1>, tensor<51x40x50x78x55x74xi1>, tensor<51x40x50x78x55x74xf32>
  }
}
