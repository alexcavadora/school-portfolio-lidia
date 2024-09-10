/*
matriz espiral
*/

#include <stdio.h>
int main()
{
	int x, y;
	scanf("%d %d", &x, &y);

	int str[x][y];
	for(int i = 0; i < x; i++)
		for (int j = 0; j < y; j++)
			str[i][j] = i + j * (j-1);

	char direction = 0; // 0 - right, 1 - down, 2 - left, 3 - up
	int curr_x = 0, curr_y = 0;
	int limit_x = 0, limit_y = 0;

	do 
	{
		switch (direction)
	{
		case 0:

			for(int i = curr_x; i <  x - limit_x; i++)
			{
				int j = curr_y;
				printf("%d",str[i][j]);
				curr_x = i;
			}
			limit_x++;
			break;
		case 1:
			for(int j = curr_y; j < y - limit_y; j++)
			{
				int i = curr_x;
				printf("%d",str[i][j]);
				curr_y = i;
			}
			limit_y++;
			break;
		case 2:
			for(int i = x - limit_x; x > 0; i--)
			{
				int j = curr_y;
				printf("%d",str[i][j]);
				curr_y = i;
			}
			limit_x++;
			break;
		case 3:
			for(int j = y - limit_y; y > 0; j--)
			{
				int i = curr_x;
				printf("%d",str[i][j]);
				curr_y = i;
			}
			limit_y++;	
			break;
	}
	if(direction < 4)
		direction++;
	else
		direction = 0;
	} while (curr_y != limit_y || curr_x != limit_x);
	
}