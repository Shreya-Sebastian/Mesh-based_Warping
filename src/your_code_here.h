#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include "helpers.h"


/*
 * Utility functions.
 */

/// <summary>
/// Bilinearly sample ImageFloat or ImageRGB using image coordinates [x,y].
/// Note that the coordinate start and end points align with the entire image and not just
/// the pixel centers (a half-pixel difference).
/// </summary>
/// <typeparam name="T">template type, can be ImageFloat or ImageRGB</typeparam>
/// <param name="image">input image</param>
/// <param name="pos">x,y position in px units</param>
/// <returns>interpolated pixel value (float or glm::vec3)</returns>
template <typename T>
inline T sampleBilinear(const Image<T>& image, const glm::vec2& pos_px)
{
    // Write a code that bilinearly interpolates values from a generic image (can contain either float or glm::vec3).
    // The pos_px input represents the (x,y) pixel coordinates of the sampled point where:
    //   [0, 0] = The left top corner of the left top (=first) pixel.
    //   [width, height] = The right bottom corner of the right bottom (=last) pixel.
    //   [0, height] = The left bottom corner of the left bottom pixel.
    // 
    // Take into account the size of individual pixel and the fact, that the "value" of pixel is conceptually stored in its center.
    //      => Example 1: For pos_px between centers of pixels, the method bilinearly interpolates between 4 nearest pixel.
    //      => Example 2: For pos_px corresponding to a center of a pixel, the method needs to return an exact value of that pixel. 
    //                    This is a natural property of any interpolation.
    // 
    // Therefore, steps are as follows:
    //     1. Determine the 4 nearest pixels.
    //     2. Bilinearly interpolate their values based on the position of the sampling point between them.
    // 
    // Note: The method is templated by parameter "T". This will be either float or glm::vec3 depending on whether the method
    // is called with ImageFloat or ImageRGB. Use either "T" or "auto" to define your variables and use glm::functions to handle both types.
    // Example:
    //    auto value = image.data[0] * 3; // both float and glm:vec3 support baisc operators
    //    T rounded_value = glm::round(image.data[0]); // glm::round will handle both glm::vec3 and float. 
    // Use glm API for further reference: https://glm.g-truc.net/0.9.9/api/a00241.html
    //   


    int x = static_cast<int>(pos_px.x);
    int y = static_cast<int>(pos_px.y);
    float fx = pos_px.x - x;
    float fy = pos_px.y - y;

    // Clamp coordinates to stay within the image bounds
    int x1 = std::clamp(x, 0, image.width - 1);
    int y1 = std::clamp(y, 0, image.height - 1);
    int x2 = std::clamp(x + 1, 0, image.width - 1);
    int y2 = std::clamp(y + 1, 0, image.height - 1);

    // Get the four nearest pixels
    auto& p11 = image.data[y1 * image.width + x1];
    auto& p12 = image.data[y1 * image.width + x2];
    auto& p21 = image.data[y2 * image.width + x1];
    auto& p22 = image.data[y2 * image.width + x2];

    // Bilinear interpolation
    T result = (1 - fx) * (1 - fy) * p11 + fx * (1 - fy) * p12 + (1 - fx) * fy * p21 + fx * fy * p22;

    return result;
}


/*
  Core functions.
*/

/// <summary>
/// Applies the bilateral filter on the given disparity image.
/// Ignored pixels that are marked as invalid.
/// </summary>
/// <param name="disparity">The image to be filtered.</param>
/// <param name="guide">The image guide used for calculating the tonal distances between pixel values.</param>
/// <param name="size">The kernel size, which is always odd.</param>
/// <param name="guide_sigma">Sigma value of the gaussian guide kernel.</param>
/// <returns>ImageFloat, the filtered disparity.</returns>
ImageFloat jointBilateralFilter(const ImageFloat& disparity, const ImageRGB& guide, const int size, const float guide_sigma)
{
    // The filter size is always odd.
    assert(size % 2 == 1);

    // We assume both images have matching dimensions.
    assert(disparity.width == guide.width && disparity.height == guide.height);

    // Rule of thumb for gaussian's std dev. 
    const float sigma = (size - 1) / 2 / 3.2f;

    // Radius of the filter kernel.
    const int radius = size / 2;

    // Empty output image.
    auto result = ImageFloat(disparity.width, disparity.height);


    //
    // Implement a bilateral filter of the disparity image guided by the guide.
    // Ignore all contributing pixels where disparity == INVALID_VALUE.
    //
    // 1. Iterate over all output pixels.
    // 2. For each output pixel, visit all neighboring pixels (including itself) in a size x size symmetric neighborhood. 
    //    That is the same as during convolution with a size x size symmetric filter centered around the pixel.
    // 3. If a neighbor is outside of the image or its disparity == INVALID_VALUE, skip the pixel (move to the next one).
    // 4. For each neighbor compute its weight as w_i = gauss(dist, sigma) * gauss(diff_value, guide_sigma)
    //      where
    //          * dist is the Eucledian distance between the center and current pixel in pixels (L2 norm).
    //          * diff_value is the L2 (euclidean) distance of the guide image pixel values for the center and current pixel.
    //          * gauss(x, sigma) is a Normal distribution pdf function for x=x and std.dev.= sigma which is available for use.
    // 5. Compute weighted mean of all (unskipped) neighboring pixel disparities and assign it to the output.
    // 
    // Refer to the first Lecture or recommended study books for more info on Bilateral filtering.
    // 
    // Notes:
    //   * If a pixel has no neighbor (all were skipped), assign INVALID_VALUE to the output.  
    //   * One point awarded for a correct OpenMP parallelization.
    // Precompute spatial Gaussian weights
    std::vector<std::vector<float>> spatialWeights(size, std::vector<float>(size));
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            // Using squared Euclidean dist --> i * i + j * j
            // Using Gaussian kernel
            spatialWeights[i + radius][j + radius] = exp(-(i * i + j * j) / (2.0f * sigma * sigma));
        }
    }

    //#pragma omp parallel for
    for (auto i = 0; i < disparity.data.size(); i++) {
            
        int x = i % disparity.width;
        int y = i / disparity.width;
        

        float K = 0.0f;
        float filteredValue = 0.0f;
        bool valid_neighbors = false;

        glm::vec3 center_guide_value = guide.data[i];

        //// Iterate through the kernel.
        for (int dx = -radius; dx <= radius; ++dx) {
            for (int dy = -radius; dy <= radius; ++dy) {
                int nx = x + dx;
                int ny = y + dy;

                // Check if the neighbor is within the image bounds
                if (nx >= 0 && nx < disparity.width && ny >= 0 && ny < disparity.height) {
                    int neighbor_index = ny * disparity.width + nx;

                    // Check if the disparity value is valid
                    if (disparity.data[neighbor_index] != INVALID_VALUE) {
                        valid_neighbors = true;

                        glm::vec3 neighbor_guide_value = guide.data[neighbor_index];

                        // Calculate the Euclidean distance in the guide image
                        float diff_value = glm::distance(center_guide_value, neighbor_guide_value);

                        // Spatial Gaussian weight (distance between the center pixel and the current pixel)
                        float dist = sqrt(dx * dx + dy * dy);
                        float spatial_weight = exp(-(dist * dist) / (2 * sigma * sigma));

                        // Range Gaussian weight (difference in guide image values)
                        float range_weight = exp(-(diff_value * diff_value) / (2 * guide_sigma * guide_sigma));

                        // Combined weight
                        float weight = spatial_weight * range_weight;

                        // Accumulate weighted sum and the sum of weights
                        filteredValue += weight * disparity.data[neighbor_index];
                        K += weight;
                    }
                }
            }
        }

        // If there were no valid neighbors, set the pixel to INVALID_VALUE
        // Otherwise, set it to the normalized filtered value
        if (valid_neighbors) {
            result.data[i] = filteredValue / K;
        } else {
            result.data[i] = INVALID_VALUE;
        }
    }
            

    //auto example = gauss(0.5f, 1.2f); // This is just an example of computing Normal pdf for x=0.5 and std.dev=1.2.


    // Return filtered disparity.
    return result;
}

/// <summary>
/// In-place normalizes and an ImageFloat image to be between 0 and 1.
/// All values marked as invalid will stay marked as invalid.
/// </summary>
/// <param name="scalar_image"></param>
/// <returns></returns>
void normalizeValidValues(ImageFloat& scalar_image)
{
    float min_val = FLT_MAX;
    float max_val = FLT_MIN;

    // Find the min and max values ignoring INVALID_VALUE
    for (auto val : scalar_image.data) {
        if (val != INVALID_VALUE) {
            if (val < min_val)
                min_val = val;
            if (val > max_val)
                max_val = val;
        }
    }

    // Rescale values to [0,1] range
    for (auto& val : scalar_image.data) {
        if (val != INVALID_VALUE) {
            val = (val - min_val) / (max_val - min_val);
        }
    }
}

/// <summary>
/// Converts a disparity image to a normalized depth image.
/// Ignores invalid disparity values.
/// </summary>
/// <param name="disparity">disparity in arbitrary units</param>
/// <returns>linear depth scaled from 0 to 1</returns>
ImageFloat disparityToNormalizedDepth(const ImageFloat& disparity)
{
    auto depth = ImageFloat(disparity.width, disparity.height);

    //
    // Convert disparity to a depth with unknown scale:
    //    depth_unscaled = 1.0 / disparity
    // If disparity of a pixel is invalid, set its depth also invalid (INVALID_VALUE).
    // We guarantee that all valid disparities > 0.
    //
        
    for (int i = 0; i < disparity.data.size(); i++) {
        float d = disparity.data[i];
        if (d == INVALID_VALUE) {
            depth.data[i] = INVALID_VALUE;
        } else {
            depth.data[i] = 1.0f / d;
        }
    }

    // Rescales valid depth values to [0,1] range.
    normalizeValidValues(depth);

    return depth;
}

/// <summary>
/// Convert linear normalized depth to target pixel disparity.
/// Invalid pixels 
/// </summary>
/// <param name="depth">Normalized depth image (values in [0,1])</param>
/// <param name="iod_mm">Inter-ocular distance in mm.</param>
/// <param name="px_size_mm">Pixel size in mm.</param>
/// <param name="screen_distance_mm">Screen distance from eyes in mm.</param>
/// <param name="near_plane_mm">Near plane distance from eyes in mm.</param>
/// <param name="far_plane_mm">Far plane distance from eyes in mm.</param>
/// <returns>screen disparity in pixels</returns>
ImageFloat normalizedDepthToDisparity(
    const ImageFloat& depth, const float iod_mm,
    const float px_size_mm, const float screen_distance_mm,
    const float near_plane_mm, const float far_plane_mm)
{
    auto px_disparity = ImageFloat(depth.width, depth.height);

    //
    // Based on physical dimensions, distance, resolution (and hence pixel size) of the screen,
    // as well as physiologically determined distance between viewers pupil (IOD or IPD),
    // compute stereoscopic pixel disparities that will make the display appear at a correct depth
    // represented by linear interpolation between the near and far plane based on the depth input image.
    // 
    // Refer to Lecture 4 for formulas.
    // 
    // Example:
    //    screen distance = 600 mm, near_plane_mm = 550, far_plane == 650, depth = 0.1  
    //         => the pixel should appear 55+0.1(65-55) = 56 cm away from the user
    //         => That is 4 cm in front of the screen.
    //         => That means the pixel disparity will be a negative number ("crossed disparity").
    // 
    // Note:
    //    * All distances are measured orthogonal on the screen and are assumed constant across the screen (ignores the eccentricity variance).
    //    * Invalid pixels (depth==INVALID_VALUE) are to be marked invalid on the output as well.
    //
    
    for (int i = 0; i < depth.data.size(); i++) {
        if (depth.data[i] == INVALID_VALUE) {
            px_disparity.data[i] = INVALID_VALUE;
        } else {
            // Convert normalized depth to actual depth z
            float normalizedDepth = depth.data[i];
            float z = near_plane_mm + normalizedDepth * (far_plane_mm - near_plane_mm) - screen_distance_mm;

            // Calculate pixel disparity
            float disparity = (iod_mm / px_size_mm) * ( z/(screen_distance_mm + z));
            px_disparity.data[i] = disparity;
        }
    }

    return px_disparity; // returns disparity measured in pixels
}

/// <summary>
/// Creates a warping grid for an image of specified height and weight.
/// It produces vertex buffer which stores 2D positions of pixel corners,
/// and index buffer which defines triangles by triplets of indices into
/// the vertex buffer (the three vertices form a triangle).
/// 
/// </summary>
/// <param name="width">Image width.</param>
/// <param name="height">Image height.</param>
/// <returns>Mesh, containing a vertex buffer and triangle index buffer.</returns>
Mesh createWarpingGrid(const int width, const int height)
{

    // Build vertex buffer.
    auto num_vertices = (width + 1) * (height + 1);
    auto vertices = std::vector<glm::vec2>(num_vertices);

    //
    // Fill the vertex buffer (vertices) with 2D coordinate of the pixel corners.
    // Expected output coordinates are:
    //   [0,0] for the left top corner of the left top (=first) pixel.
    //   [width,height] for the right bottom corner of the right bottom (=last) pixel.
    //   [0,height] for the left bottom corner of the left bottom pixel.
    // 
    // The order in memory is to be the same as for images (row after row).
    // 
    
    for (int y = 0; y <= height; ++y) {
        for (int x = 0; x <= width; ++x) {
            vertices[y * (width + 1) + x] = glm::vec2(x, y);
        }
    }

    // Build index buffer.
    auto num_pixels = width * height;
    auto num_triangles = num_pixels * 2;
    auto triangles = std::vector<glm::ivec3>(num_triangles);

    //
    // Fill the index buffer (triangles) with indices pointing to the vertex buffer.
    // Each element of the "triangles" is an integer triplet (glm::ivec3). 
    // It represents a triangle by selecting 3 vertices from the vertex buffer defining its corners in
    // a clockwise manner.
    // We need to fill the index buffer in the same order as pixels are stored in memory (that is row by row)
    // and for each pixel we should generate two triangles that together cover the are of a pixel as follows:
    // 
    //   A ------- B
    //   |  *      |
    //   |    *    |
    //   |      *  |
    //   D ------- C
    // 
    // Where A,B,C,D are the CORNERS of the respective pixel.
    // 
    // For each such pixel, we add two triangles: 
    //     glm::ivec3(A,B,C) and glm::ivec3(A,C,D)  (in this exact order)
    // where A,B,C,D are indices into the vertex buffer.
    // 
    // The result should be a grid that fills an entire image and replaces each pixel with two small triangles.
    //
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int base_index = y * (width + 1) + x; // Index of the top-left corner of the current pixel
            int triangle_index = (y * width + x) * 2; // Index in the triangle list

            // First triangle (top-left triangle)
            triangles[triangle_index] = glm::ivec3(
                base_index, // A
                base_index + 1, // B
                base_index + width + 2 // C
            );

            // Second triangle (bottom-right triangle)
            triangles[triangle_index + 1] = glm::ivec3(
                base_index, // A
                base_index + width + 2, // C
                base_index + width + 1 // D
            );
        }
    }

    // Combine the vertex and index buffers into a mesh.
    return Mesh { std::move(vertices), std::move(triangles) };
}

/// <summary>
/// Warps a grid based on the given disparity and scaling_factor.
/// </summary>
/// <param name="grid">The original mesh.</param>
/// <param name="disparity">Disparity for each PIXEL.</param>
/// <param name="scaling_factor">Global scaling factor for the disparity.</param>
/// <returns>Mesh, the warped grid.</returns>
Mesh warpGrid(Mesh& grid, const ImageFloat& disparity, const float scaling_factor, const BilinearSamplerFloat& sampleBilinear)
{
    // Create a copy of the input mesh (all values are copied).
    auto new_grid = Mesh { grid.vertices, grid.triangles };

    const float EDGE_EPSILON = 1e-5f * disparity.width;

    //
    // The goal is to modify the x coordinate of the grid vertices based on 
    // the scaled pixel disparity corresponding to the original location of the vertex buffer:
    // 
    // new_grid.vertex.x = grid.vertex.x + scaling_factor * sampled_disparity
    // 
    // where sampled_disparity is a bilinearly interpolated value from the disparity image
    // which we can easily obtained using our other function "sampleBilinear":
    //     sampled_disparity = sampleBilinear(disparity, grid.vertex)
    // 
    // IMPORTANT - in order to keep the grid attached to the image "frame",
    // we must not move the border vertices away => do not move vertices
    // which are within EDGE_EPSILON from the left/right image boundary in the x direction.
    // That means the grid is attached to left and right boundaries but can freely slide along the top/down boundaries.
    //

    // Here is an example use of the bilinear interpolation (using the provided function argument).
    //auto interpolated_value = sampleBilinear(disparity, glm::vec2(1.0f, 1.0f));
    // Recommended test: For a 2x2 image it SHOULD return the mean of the 4 pixels.
    
    for (size_t i = 0; i < grid.vertices.size(); ++i) {
        float x = grid.vertices[i].x;
        float y = grid.vertices[i].y;

        // Ensure boundary vertices are not moved (left and right edges)
        if (x <= EDGE_EPSILON || x >= disparity.width - EDGE_EPSILON) {
            continue; // Skip warping for edge vertices
        }

        // Sample the disparity map at the vertex position
        glm::vec2 pos(x, y);
        float sampled_disparity = sampleBilinear(disparity, pos);

        // Apply the disparity scaling factor to compute new x position
        new_grid.vertices[i].x = x + scaling_factor * sampled_disparity;
    }




    return new_grid;
}



/// <summary>
/// Forward-warps an image based on the given disparity and warp_factor.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="disparity">Disparity of the source image in pixels.</param>
/// <param name="warp_factor">Multiplier of the disparity.</param>
/// <returns>ImageWithMask, containing the forward-warped image and a mask image. Mask=1 for valid pixels, Mask=0 for holes</returns>
ImageWithMask forwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const ImageFloat& disparity, const float warp_factor)
{
    // The dimensions of src image, src depth and disparity maps all match.
    assert(src_image.width == disparity.width && src_image.height == disparity.height);
    assert(src_image.width == disparity.width && src_depth.height == src_depth.height);
    
    // Create a new image and depth map for the output.
    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_mask = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth mask map with zero.
    std::fill(dst_mask.data.begin(), dst_mask.data.end(), 0.0f);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth map with a very large number.
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), std::numeric_limits<float>::max());

    // 
    // The goal is to forward warp the image pixels using the disparity displacement provided in 
    // the disparity map along with additional scaling factor. Furthermore, we
    // use depth information to resolve conflicts when multiple pixels attempt
    // to write a single output pixel. 
    //
    // 1. For every input pixel, compute where it should be warped to 
    //    based on the associated disparity and warp_factor. 
    //    Use standard rounding rules to obtain integer position (ie., 0.5 rounds up).
    //      x' = rounding_function(x + disparity * warp_factor)
    //      y' = y
    // 
    // 2. Check the destination depth at the [x',y'] location and compare it with the 
    //    depth of the currently warped pixel (ie., depth[x,y]).
    //
    // 3. If the currently warped pixel has a depth larger or equal to the previous value in the output depth buffer,
    //    stop here and continue with step 1 for the next pixel.
    //
    // 4. Overwrite the output buffers. This means writing:
    //    - the destination image
    //    - the destination depth map
    //    - the mask (mask->1)
    //
    // 
    // Note: Point(s) awarded for a correct and efficient parallel solution using OpenMP.
    //
    
    for (int y = 0; y < src_image.height; ++y) {
        for (int x = 0; x < src_image.width; ++x) {
            int src_index = y * src_image.width + x;
            float d = disparity.data[src_index] * warp_factor; // Apply warp factor to disparity
            float src_z = src_depth.data[src_index];

            // Calculate the new x coordinate after applying the disparity
            int dst_x = std::round(x + d);
            int dst_y = y; // The y coordinate remains unchanged

            // Ensure the destination coordinates are within the image bounds
            if (dst_x >= 0 && dst_x < src_image.width && dst_y >= 0 && dst_y < src_image.height) {
                int dst_index = dst_y * src_image.width + dst_x;

                // Depth test to resolve conflicts: only overwrite if the source depth is less than the current depth at the destination
                if (src_z < dst_depth.data[dst_index]) {
                    dst_image.data[dst_index] = src_image.data[src_index];
                    dst_depth.data[dst_index] = src_z;
                    dst_mask.data[dst_index] = 1.0f; // Mark this pixel as valid
                }
            }
        }
    }

    // Return the warped image.
    return ImageWithMask(dst_image, dst_mask);

}


/// <summary>
/// Applies the bilateral filter on the given image to fill the holes
/// indicated by a binary mask (mask==0 -> missing pixel).
/// Keeps the pixels not marked as holes unchanged.
/// </summary>
/// <param name="img_forward">The image to be filtered and its mask.</param>
/// <param name="size">The kernel size. It is always odd.</param>
/// <param name="guide_sigma">Sigma value of the gaussian guide kernel.</param>
/// <returns>ImageRGB, the filtered forward warping image.</returns>
ImageRGB inpaintHoles(const ImageWithMask& img, const int size)
{
    // The filter size is always odd.
    assert(size % 2 == 1);

    // Rule of thumb for gaussian's std dev.
    const float sigma = (size - 1) / 2 / 3.2f;
    const int radius = size / 2;
    // The output is initialized by copy of the input.
    auto result = ImageRGB(img.image);
    

    // The goal is to fill the holes in forward warping image using a bilateral filter where the mask serves as a guide.
    
    // 1. For valid pixels (mask >= 0.5) -> do nothing and keep the valid pixel.
    // 2. For invalid pixels -> replace pixel with:
    //      Filter valid neighbors in [size x size] in a symmetric neighborhood 
    //      with a Gaussian filter weight w_i = gauss(dist, sigma) where
    //          * dist is the Euclidean distance between the center and current pixel in pixels (L2 norm).
    //          * gauss(x, sigma) is a Normal distribution pdf function for x=x and std.dev.= sigma which is available for use.
    //      !!!! SKIP invalid neighbors (mask < 0.5) !!!!
    // 
    // Notes:
    //   * Keep pixels with no valid neighbors unmodified.

    for (int y = 0; y < img.image.height; ++y) {
        for (int x = 0; x < img.image.width; ++x) {
            int idx = y * img.image.width + x;

            // Skip valid pixels (mask >= 0.5)
            if (img.mask.data[idx] >= 0.5) {
                continue; // Keep the valid pixel unchanged
            }

            float sumWeights = 0.0f;
            float rSum = 0.0f, gSum = 0.0f, bSum = 0.0f;
            bool hasValidNeighbors = false;

            // Check the neighborhood for valid pixels to apply the Gaussian filter
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;

                    // Ensure within image bounds
                    if (nx >= 0 && nx < img.image.width && ny >= 0 && ny < img.image.height) {
                        int neighborIdx = ny * img.image.width + nx;

                        // Skip invalid neighbors (mask < 0.5)
                        if (img.mask.data[neighborIdx] < 0.5) {
                            continue;
                        }

                        hasValidNeighbors = true;
                        float dist = sqrt(dx * dx + dy * dy);
                        float weight = gauss(dist, sigma);

                        rSum += weight * img.image.data[neighborIdx].r;
                        gSum += weight * img.image.data[neighborIdx].g;
                        bSum += weight * img.image.data[neighborIdx].b;
                        sumWeights += weight;
                    }
                }
            }

            // Apply the filtered value if there are valid neighbors
            if (hasValidNeighbors && sumWeights > 0) {
                result.data[idx].r = static_cast<uint8_t>(rSum / sumWeights);
                result.data[idx].g = static_cast<uint8_t>(gSum / sumWeights);
                result.data[idx].b = static_cast<uint8_t>(bSum / sumWeights);
            }
            // Else, keep the pixel unchanged if no valid neighbors were found
        }
    }


    // Return inpainted image.
    return result;
}

/// <summary>
/// Backward-warps an image using a warped mesh.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="src_grid">Source grid.</param>
/// <param name="dst_grid">The warped grid.</param>
/// <param name="sampleBilinear">a function that bilinearly samples ImageFloat</param>
/// <param name="sampleBilinearRGB">a function that bilinearly samples ImageRGB</param>
/// <returns>ImageRGB, the backward-warped image.</returns>
ImageRGB backwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const Mesh& src_grid, const Mesh& dst_grid,
    const BilinearSamplerFloat& sampleBilinear, const BilinearSamplerRGB& sampleBilinearRGB)
{
    // The dimensions of src image and depth match.
    assert(src_image.width == src_depth.width && src_image.height == src_depth.height);
    // We assume that both grids have the same size and also the same order (ie., there is 1:1 triangle pairing).
    // This implies that the content of index buffers of both meshes are exactly equal (we do not test it here).
    assert(src_grid.triangles.size() == dst_grid.triangles.size());

    // Create a new image and depth map for the output.
    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth map with a very large number.
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), 1e20f);

    //
    // This method implements mesh-based warping by rasterizing
    // each triangle of the destination grid and sampling
    // the source texture by looking up corresponding 
    // position in the source grid using barycentric coordinates.
    // 
    // 1. For every triangle in the warped grid (dst_grid), 
    //    determine X and Y ranges of destination pixel coordinates
    //    whose **centers** lie in the bounding box of the triangle.
    //    Note: Look back to createWarpingGrid() for definition
    //          of the grid vertex coordinates.
    // 
    // 2. Enumerate all candidate pixels within the bounding box.
    //    Skip pixels whose centers actually do not lie inside the triangle.
    //    Use the provided isPointInsideTriangle(pt_dst, vert_a, vert_b, vert_c)
    //    method to do the test.
    // 
    // 3. Compute barycentric coordinates of the pixel center in the triangle
    //    using the provided method bc = barycentricCoordinates(pt_dst, vert_a, vert_b, vert_c).
    // 
    // 4. Find the corresponding triangle in the original mesh. It has the same triangle index.
    // 
    // 5. Reproject the pixel center to the source grid by using the already computed barycentric
    //    coordinates:
    //        pt_src = (vert_a_src, vert_b_src, vert_c_src) * bc
    // 
    // 6. Bilinearly sample both the source depth map and the source image values at the pt_src
    //    position. 
    //    Hint: Use the sampleBilinear[RGB]() method implemented earlier (using the provided function arguments).
    // 
    // 7. Compare the depth in the source and destination depth values and implement the depth-test
    //    like in fowardWarpImage().
    //    Hint: The destination map can be accessed directly without interpolation because we are
    //          computing for an exact pixel coordinate.
    // 
    // 8. If the depth test passes, write out the destination image (dst_image) and depth (dst_depth) values. 
    //    Again, it is the same logic as in fowardWarpImage().
    // 
    // 9. Return the warped image (dst_image).
    //
    
    // Example of testing point [0.1, 0.2] is inside a triangle.
    bool is_point_inside = isPointInsideTriangle(glm::vec2(0.1, 0.2), glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0, 1));

    // Example of computing barycentric coordinates of a point [0.1, 0.2] inside a triangle.
    glm::vec3 bc = barycentricCoordinates(glm::vec2(0.1, 0.2), glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0, 1));

    //
    //    YOUR CODE GOES HERE
    //

    // Return the warped image.
    return dst_image;
}

/// <summary>
/// Returns an anaglyph image.
/// </summary>
/// <param name="image_left">left RGB image</param>
/// <param name="image_right">right RGB image</param>
/// <param name="saturation">color saturation to apply</param>
/// <returns>ImageRGB, the anaglyph image.</returns>
ImageRGB createAnaglyph(const ImageRGB& image_left, const ImageRGB& image_right, const float saturation)
{
    // An empty image for the resulting anaglyph.
    auto anaglyph = ImageRGB(image_left.width, image_left.height);

    // 
    // Convert stereoscopic pair into a single anaglyph stereoscopic image
    // for viewing in red-cyan anaglyph glasses.
    // We additionally scale saturation of the image to make the image
    // more "grayscale" since colors are problematic in analglyph image
    // and increase crosstalk (ghosting).
    // 
    // For both left and rigt image:
    // 1. Convert RGB to HSV color space using the provided rgbToHsv() function.
    // 2. Scale the saturation (stored in the second (=Y) component of the vec3) by the "saturation" param.
    // 3. Convert back to RGB using the hsvToRgb().
    // 
    // Combine the two images such that:
    //    * output.red = left.red
    //    * output.green = right.green
    //    * output.blue = right.blue.
    //

    // Example: RGB->HSV->RGB should be approx identity.
    //auto rgb_orig = glm::vec3(0.2, 0.6, 0.4);
    //auto rgb_should_be_same = hsvToRgb(rgbToHsv(rgb_orig)); // expect rgb == rgb_2 (up to numerical precision)

    for (int y = 0; y < image_left.height; ++y) {
        for (int x = 0; x < image_left.width; ++x) {
            int index = y * image_left.width + x;

            // Convert RGB to HSV
            glm::vec3 hsv_left = rgbToHsv(image_left.data[index]);
            glm::vec3 hsv_right = rgbToHsv(image_right.data[index]);

            // Scale the saturation
            hsv_left.y *= saturation;
            hsv_right.y *= saturation;

            // Convert back to RGB
            glm::vec3 rgb_left = hsvToRgb(hsv_left);
            glm::vec3 rgb_right = hsvToRgb(hsv_right);

            // Combine to form anaglyph
            anaglyph.data[index].r = rgb_left.r;
            anaglyph.data[index].g = rgb_right.g;
            anaglyph.data[index].b = rgb_right.b;
        }
    }

    // Returns a single analgyph image.
    return anaglyph;
}


/// <summary>
/// Rotates a grid counter-clockwise around the center by a given angle in degrees.
/// </summary>
/// <param name="grid">The original mesh.</param>
/// <param name="center">The center of the rotation (in pixel coords).</param>
/// <param name="angle">Angle in degrees.</param>
/// <returns>Mesh, the rotated grid.</returns>
Mesh rotatedWarpGrid(Mesh& grid, const glm::vec2& center, const float& angle)
{
    // Create a copy of the input mesh (all values are copied).
    auto new_grid = Mesh { grid.vertices, grid.triangles };

    const float DEGREE2RADIANS = 0.0174532925f;

    //
    // The goal is to rotate the coordinate of the grid vertices
    // counter-clockwise around the 'center' by a given angle in degrees:
    //
    // 1. Create a 3*3 matrix T, there are three steps for this matrix:
    //    Translate from origin to center,
    //    then rotate counter-clockwise by a given angles,
    //    and translate back from center to origin.
    //
    // 2. Multiply each vertex by matrix T to get its new coordinates.
    //

    //
    //    YOUR CODE GOES HERE
    //


    return new_grid;
}



/// <summary>
/// Rotate an image using backward warping based on the provided meshes.
/// </summary>
/// <param name="image">input image</param>
/// <param name="src_grid">original grid</param>
/// <param name="dst_grid">rotated grid</param>
/// <param name="sampleBilinear">a function that bilinearly samples ImageFloat</param>
/// <param name="sampleBilinearRGB">a function that bilinearly samples ImageRGB</param>
/// <returns>rotated image, has the same size as input</returns>
ImageRGB rotateImage(const ImageRGB& image, const Mesh& src_grid, const Mesh& dst_grid, const BilinearSamplerFloat& sampleBilinear, const BilinearSamplerRGB& sampleBilinearRGB)
{

    //
    // Unused pixels should be black.
    // Pixel that fall outside of the image should be discarded.
    //
    //    YOUR CODE GOES HERE
    //
    return ImageRGB(1, 1); // replace
}